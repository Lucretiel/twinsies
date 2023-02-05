/*!
Twinsies is a special shared pointer, similar to an [`Arc`], where two specific
objects (called [`Joint`]) share joint ownership of the underlying object. The
key difference compared to an [`Arc`] is that the underlying object is dropped
when *either* of the [`Joint`] objects go out of scope.

Because a single [`Joint`] cannot, by itself, keep the shared object alive, it
cannot be dereferenced directly like an [`Arc`]. Instead, it must be locked
with [`.lock()`]. While locked, the object is guaranteed to stay alive as long
as the [`JointLock`] is alive. If the a [`Joint`] is dropped while its partner
is locked, the object stays alive, but it dropped immediately as soon as the
other [`Joint`] is no longer locked.

Twinsies is intended to be used for things like unbuffered channels, join
handles, and async [`Waker`]- cases where some piece of shared state should
only be preserved as long as *both* halves are still interested in it.

# Example

```rust
use twinsies::Joint;
use std::cell::Cell;

let (first, second) = Joint::new(Cell::new(0));

assert_eq!(first.lock().unwrap().get(), 0);

first.lock().unwrap().set(10);
assert_eq!(second.lock().unwrap().get(), 10);

drop(second);

// Once `second` is dropped, the shared value is gone
assert!(first.lock().is_none())
```

# Locks preserve liveness
```
use twinsies::Joint;
use std::cell::Cell;

let (first, second) = Joint::new(Cell::new(0));

let lock = first.lock().unwrap();

lock.set(10);

assert_eq!(second.lock().unwrap().get(), 10);
second.lock().unwrap().set(20);

assert_eq!(lock.get(), 20);

drop(second);

assert_eq!(lock.get(), 20);
lock.set(30);
assert_eq!(lock.get(), 30);

// As soon as the lock is dropped, the shared value is gone, since `second`
// was dropped earlier
drop(lock);
assert!(first.lock().is_none());
```

[`Arc`]: std::sync::Arc
[`Weak`]: std::sync::Weak
[`Waker`]: std::task::Waker
[`.lock()`]: Joint::lock
*/

extern crate alloc;

use std::{
    cell::UnsafeCell,
    fmt::{self, Debug, Formatter},
    hint::unreachable_unchecked,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::Deref,
    process::abort,
    ptr::NonNull,
    sync::atomic::{AtomicU32, Ordering},
};

use alloc::boxed::Box;

/// Identical to `unreachable`, but panics in debug mode. Still requires unsafe.
macro_rules! debug_unreachable {
    ($($arg:tt)*) => {
        match cfg!(debug_assertions) {
            true => unreachable!($($arg)*),
            false => unreachable_unchecked(),
        }
    }
}

const MAX_COUNT: u32 = i32::MAX as u32;

struct JointContainer<T> {
    // It's not clear to me if we actually need an `UnsafeCell` here, but better
    // safe then sorry. The key issue at play is that multiple threads might
    // hold an &JointContainer<T> while this is being manually dropped.
    //
    // We prefer to use a `MaybeUninit` instead of a `ManuallyDrop`, because the
    // value could exist in an uninitialized state for a while (while only one
    // Joint exists), so we want to make it unsafe to get a reference to it.
    value: UnsafeCell<MaybeUninit<T>>,

    // *In general*, this counts the number of existing handles (joints +
    // locks). The exception to this rule is that, when a drop reduces the count
    // to 1, that drops the value, then *immediately* attempts to decrement the
    // count down to 0. Summary of states:
    //
    // - 0: When we observe a 0, it means that this is the last Joint in
    //   existence and that the value was previously dropped. New lock attempts
    //   will fail and we can drop the container when we drop.
    // - 1: When we decrement to 1, it means that either this is one of the two
    //   joints, or that this is a lock and the other joint dropped while we
    //   existed. In either case, it means that we can drop the value. We then
    //   immediately attempt to decrement the count to 0; if we succeed, the
    //   last joint will take care of dropping the container, otherwise, we need
    //   to drop the container ourselves, because the last joint dropped while
    //   we were dropping the value
    // - 2+: either both joints exist, or a joint exists and is locked. In
    //   either case, the value is alive, and becomes dead when the count drops
    //   to 1
    count: AtomicU32,
}

impl<T> JointContainer<T> {
    /// Drop the stored value. This method should only be called when only one
    /// joint exists, and it's unlocked. You must ensure that the value is
    /// never accessed after this method is called.
    #[inline]
    pub unsafe fn drop_value_in_place(&self) {
        self.value
            .get()
            .as_mut()
            .expect("UnsafeCell shouldn't return a null pointer")
            .assume_init_drop()
    }

    /// Assume that the value hasn't been dropped yet and get a reference to it.
    /// You must not call this if the value has been dropped in place.
    #[inline]
    #[must_use]
    pub unsafe fn get_value(&self) -> &T {
        self.value
            .get()
            .as_ref()
            .expect("UnsafeCell shouldn't return a null pointer")
            .assume_init_ref()
    }
}

/// A thread-safe shared ownership type that shares ownership with a partner, such
/// that the shared object is dropped when *either* [`Joint`] goes out of scope.
///
/// See [module docs][crate] for details.
pub struct Joint<T> {
    container: NonNull<JointContainer<T>>,
    phantom: PhantomData<JointContainer<T>>,
}

unsafe impl<T: Send + Sync> Send for Joint<T> {}
unsafe impl<T: Send + Sync> Sync for Joint<T> {}

impl<T> Joint<T> {
    // Note that, while it's guaranteed that the container exists, it's not
    // guaranteed that the value is in an initialized state.
    //
    // This function on its own is always safe to call, since the container
    // exists until *all* joints are dropped.
    #[inline]
    #[must_use]
    fn container(&self) -> &JointContainer<T> {
        unsafe { self.container.as_ref() }
    }

    /// Create a new pair of `Joint`s, which share ownership of a value. When
    /// *either* of these joints is dropped, the shared value will be dropped
    /// immediately.
    #[must_use]
    #[inline]
    pub fn new(value: T) -> (Self, Self) {
        let container = Box::new(JointContainer {
            value: UnsafeCell::new(MaybeUninit::new(value)),
            count: AtomicU32::new(2),
        });

        let container = NonNull::new(Box::into_raw(container)).expect("box is definitely non null");

        (
            Joint {
                container,
                phantom: PhantomData,
            },
            Joint {
                container,
                phantom: PhantomData,
            },
        )
    }

    /// Attempt to get a reference to the stored value. This only succeeds if
    /// both joints still exist, or if this joint is already locked. The shared
    /// value is guaranteed to exist as long as the lock exists, even if the
    /// other joint is dropped.
    #[must_use]
    pub fn lock(&self) -> Option<JointLock<'_, T>> {
        let count = &self.container().count;

        let mut current = count.load(Ordering::Relaxed);

        loop {
            // We can only lock this if *both* handles currently exist. TODO:
            // prevent the distribution of new locks after the other handle has
            // dropped (currently, if this handle has some outstanding locks, it
            // may create more). In general we're not worried because the
            // typical usage pattern is that each joint will only ever make 1
            // lock at a time.
            current = match current {
                // The other `Joint` dropped (or is in the middle of being
                // dropped), so we can no longer create new locks.
                0 | 1 => break None,

                // There are too many locks already, probably because they're
                // being leaked.
                //
                // We abort here because, to quote `Arc` (which does the same
                // thing): "this is such a degenerate scenario that we don't
                // care about what happens -- no real program should ever
                // experience this."
                n if n > MAX_COUNT => abort(),

                // Increasing the reference count can always be done with Relaxed– New
                // references to an object can only be formed from an existing
                // reference, and passing an existing reference from one thread to
                // another must already provide any required synchronization.
                // n >= 2, so the object is alive.
                n => match count.compare_exchange_weak(
                    n,
                    n + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        break Some(JointLock {
                            container: self.container(),
                        })
                    }
                    Err(n) => n,
                },
            }
        }
    }

    /// Check to see if the underlying object is alive. This requires either
    /// that the other [`Joint`] still exists or that this one is currently
    /// locked.
    ///
    /// Note that another thread can cause this to become false at any time.
    /// However, once this returns false, it will never again return true for
    /// this specific [`Joint`] instance.
    #[inline]
    #[must_use]
    pub fn alive(&self) -> bool {
        self.container().count.load(Ordering::Relaxed) >= 2
    }
}

impl<T> Debug for Joint<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.container().count.load(Ordering::Relaxed) {
            0 | 1 => write!(f, "Joint(<unpaired>)"),

            // Technically it could be unpaired but still have live locks.
            // We're not really worried about that case.
            _ => write!(f, "Joint(<paired>)"),
        }
    }
}

impl<T> fmt::Pointer for Joint<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.container, f)
    }
}

impl<T> Drop for Joint<T> {
    fn drop(&mut self) {
        let mut current = self.container().count.load(Ordering::Acquire);

        /*
        Author note: there's something I'm not understanding about some of the
        atomic orderings here. As near as I can tell from my own analysis:

        - It should be sufficient that the first compare-exchange here is
          acquire/acquire (because joints themselves can't do any interesting
          writes with the shared value, and JointLocks *do* release write the
          reference count)
        - It should be sufficient that the second compare-exchange here is
          release/relaxed, because *this* thread is doing the drop with exclusive
          access, which means it can't possibly acquire any additional
          interesting writes from other joints

        However, miri complains if I downgrade the orderings past this point.
        For now I'm going to just add speculation to this comment block until
        I can pin down for sure what's going on.

        - I think a Release write might be necessary in the *first*
          compare-exchange because otherwise the acquire might not work. The
          basic question is about this flow:
          release-store 2 (the joint lock)
          relaxed-store 1 (the joint itself)

          if we acquire-load the 1, does that acquire changes released by the 2?
          if not, then the store of the 1 needs to be release.
        */
        loop {
            current = match current {
                // The value has been fully dropped, this is the last remaining
                // handle in existence. Whichever handle stored the 0 (either
                // our child lock or the other Joint) did so expecting us to
                // drop the container itself, so do so now.
                0 => {
                    drop(unsafe { Box::from_raw(self.container.as_ptr()) });
                    return;
                }

                n => match self.container().count.compare_exchange_weak(
                    n,
                    n - 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    // All failures, spurious or otherwise, need to be retried.
                    // There's no "fast escape" case (like there are in other
                    // compare-exchange sites) because we always need to ensure
                    // that n - 1 was stored.
                    Err(n) => n,

                    // Can't possibly have replaced a 0; we check for that case
                    // before attempting the compare-exchange.
                    Ok(0) => unsafe { debug_unreachable!() },

                    // The other joint is in the middle of dropping the value.
                    // We stored a 0, so it will also take care of dropping the
                    // container itself.
                    Ok(1) => return,

                    // The other joint exists and isn't locked, which means it's
                    // time to drop the value. After we finish dropping the
                    // value, we'll try to store a 0 (indicating that the other
                    // Joint should drop the container itself) or else load a 0,
                    // indicating that the other Joint dropped while we were
                    // dropping the value, so we *also* need to drop the
                    // container.
                    Ok(2) => {
                        unsafe { self.container().drop_value_in_place() }

                        // At this point we need to release store the 0, to
                        // ensure our drop propagates to other threads. We did
                        // the drop, so there's no other changes we might need
                        // to acquire. If we find there's already a zero, the
                        // other joint dropped while we were dropping value, so
                        // we also handle dropping the container.

                        match self.container().count.compare_exchange(
                            1,
                            0,
                            Ordering::Release,
                            Ordering::Acquire,
                        ) {
                            // We stored a zero; the other Joint will be
                            // responsible for deallocating the container
                            Ok(_) => return,

                            // There was already a 0; the other joint
                            // dropped while we were dropping the value.
                            // Deallocate.
                            //
                            // We apparently need to acquire-load this 0 and to
                            // be honest I have no idea why. The only way we're
                            // at this point is if this thread, moments ago,
                            // called drop_value_in_place, so what other changes
                            // could we possibly need to acquire from other
                            // threads? I guess maybe we need to acquire that
                            // the other joint is entirely done dropping itself
                            // so that the box drop here doesn't occur while
                            // the other joint's reference to `count` still
                            // exists?
                            Err(0) => {
                                drop(unsafe { Box::from_raw(self.container.as_ptr()) });
                                return;
                            }

                            // Spurious failure
                            Err(1) => unsafe {
                                debug_unreachable!(
                                    "Spurious failure in compare_exchange \
                                    should be impossible"
                                )
                            },

                            // It's never possible for the count to
                            // transition from 1 to any value other than 0
                            // or 1.
                            Err(n) => unsafe {
                                debug_unreachable!(
                                    "Joint count became {n} \
                                        after it previously stored 1"
                                )
                            },
                        }
                    }

                    // The other joint exists and is locked, which means it
                    // will take care of dropping the value.
                    Ok(_) => return,
                },
            }
        }
    }
}

impl<T> Unpin for Joint<T> {}

/// A lock associated with a [`Joint`], providing shared access to the
/// underlying value.
///
/// This object provides [`Deref`] access to the underlying shared object. It
/// guarantees that the shared object stays alive for at least as long as the
/// lock itself does, even if the other [`Joint`] is dropped.
///
/// See [module docs][crate] for details.
pub struct JointLock<'a, T> {
    container: &'a JointContainer<T>,
}

impl<T> JointLock<'_, T> {
    // It's convenient for various reasons to store a reference in the
    // `JointLock` itself and only get a raw pointer if we really need one.
    #[inline]
    #[must_use]
    fn pointer_to_container(&self) -> NonNull<JointContainer<T>> {
        NonNull::from(self.container)
    }
}

// Theoretically we could *not* add `Send` or `Sync` to the lock type; this
// loosen ordering restrictions on its drop implementation, since we could
// guarantee it stayed in the same thread as its parent. However, that would
// preclude its use in certain convenient cases (like in rayon, or across await
// boundaries in Send async functions), so we add them anyway.
unsafe impl<T: Send + Sync> Send for JointLock<'_, T> {}
unsafe impl<T: Send + Sync> Sync for JointLock<'_, T> {}

impl<T> Deref for JointLock<'_, T> {
    type Target = T;

    #[inline]
    #[must_use]
    fn deref(&self) -> &Self::Target {
        // Safety: if a JointLock exists, it's guaranteed that the value will
        // be alive for at least the duration of the lock
        unsafe { self.container.get_value() }
    }
}

impl<T: Debug> Debug for JointLock<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T> fmt::Pointer for JointLock<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.pointer_to_container(), f)
    }
}

impl<T> Clone for JointLock<'_, T> {
    fn clone(&self) -> Self {
        // The logic for cloning a joint lock can be a bit simpler than for
        // locking a joint, because we're guaranteed that the value is alive
        // (because this lock exists presently)
        //
        // Much like with lock, we can do a relaxed increment. See Joint::lock
        // for details.
        let old_count = self.container.count.fetch_add(1, Ordering::Relaxed);

        if old_count > MAX_COUNT {
            abort()
        }

        JointLock {
            container: self.container,
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        if self.pointer_to_container() != source.pointer_to_container() {
            *self = source.clone()
        }
    }
}

impl<T> Drop for JointLock<'_, T> {
    fn drop(&mut self) {
        let count = &self.container.count;

        // The logic here can be a little simpler than Joint, because we're
        // guaranteed that there's at least one other handle in existence (our
        // parent), and that it definitely won't be dropped before we're done
        // being dropped (because we've borrowed it)
        // - Need to acquire any changes made by other threads before dropping
        // - Need to release any changes made by *this* thread so that it can be
        //   dropped by another thread.
        match count.fetch_sub(1, Ordering::AcqRel) {
            // The count must be at LEAST 2, before the subtract: one for us and
            // one for our parent
            n @ (0 | 1) => unsafe {
                debug_unreachable!(
                    "Joint count was {n} while dropping a \
                    JointLock; this shouldn't be possible"
                )
            },

            // If the count was 2, it means that the other joint dropped while
            // this lock existed. We've already stored the 1, which means we've
            // taken responsibility for attempting to drop (and that future
            // attempts to lock will now fail)
            2 => {
                unsafe { self.container.drop_value_in_place() }

                // Now that the drop is finished, we can store a 0, so that our
                // parent Joint knows to drop the container itself. There's no
                // need at this point to compare-exchange, since we're
                // guaranteed that the other joint is gone and that our parent
                // joint won't drop before we're done dropping ourselves.
                // We release-store the 0 so that the drop-in-place is visible
                // to our parent.
                count.store(0, Ordering::Release)
            }

            // If the count was higher than two, the value is still alive even
            // after this lock drops
            _ => {}
        }
    }
}

impl<T> Unpin for JointLock<'_, T> {}

#[cfg(test)]
mod tests {
    use std::{hint::black_box, sync, thread};

    use crate::Joint;

    #[test]
    fn drop_test() {
        struct Container(sync::Mutex<Vec<i32>>);

        impl Drop for Container {
            fn drop(&mut self) {
                let data = self.0.get_mut().unwrap_or_else(|err| err.into_inner());
                let data = black_box(data);
                data.push(5);
                println!("{data:?}");
            }
        }

        for i in 0..100 {
            let barrier = sync::Barrier::new(2);
            let barrier = &barrier;

            let (joint1, joint2) = Joint::new(Container(sync::Mutex::new(Vec::new())));

            thread::scope(move |s| {
                let thread1 = s.spawn(move || {
                    barrier.wait();

                    if let Some(lock) = joint1.lock() {
                        lock.0.lock().unwrap_or_else(|e| e.into_inner()).push(i * 2);
                    }
                });

                let thread2 = s.spawn(move || {
                    barrier.wait();

                    if let Some(lock) = joint2.lock() {
                        lock.0.lock().unwrap_or_else(|e| e.into_inner()).push(i * 3);
                    }
                });

                thread1.join().unwrap();
                thread2.join().unwrap();
            })
        }
    }
}
