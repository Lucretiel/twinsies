#![no_std]

extern crate alloc;

use core::{
    cell::UnsafeCell,
    fmt::{self, Debug, Formatter},
    hint::unreachable_unchecked,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::Deref,
    ptr::NonNull,
    sync::atomic::{AtomicU8, Ordering},
};

use alloc::boxed::Box;

/// Identical to `unreachable!`, but only in debug mode. In release mode this
/// will be unreachable_unchecked. It therefore requires unsafe to use. Be
/// extra careful about invariants!
macro_rules! debug_unreachable {
    ($($arg:tt)*) => {
        match cfg!(debug_assertions) {
            true => unreachable!($($arg)*),
            false => unreachable_unchecked()
        }
    }
}

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
    // locks). A joint can be locked at most once (enforced by mutable borrow),
    // so this can be at most 4. The exception to this rule is that, when a drop
    // reduces the count to 1, that drops the value, then *immediately* attempts
    // to decrement the count down to 0. Summary of states:
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
    // - 2: either both joints exist, or a joint exists and was locked when the
    //   other joint dropped. Either way, value is alive and well, but will be
    //   dropped when the joint or lock is dropped.
    // - 3: both joins exist and one of them is locked. Nothing interesting is
    //   hapenning here.
    // - 4: both joins exist and both are locked.
    count: AtomicU8,
}

impl<T> JointContainer<T> {
    /// Drop the stored value. This method should only be called when only one
    /// joint exists, and it's unlocked. You must ensure that the value is
    /// never accessed after this method is called.
    pub unsafe fn drop_value_in_place(&self) {
        self.value
            .get()
            .as_mut()
            .expect("UnsafeCell shouldn't return a null pointer")
            .assume_init_drop()
    }

    /// Assume that the value hasn't been dropped yet and get a reference to it.
    /// You must not call this if the value has been dropped in place.
    pub unsafe fn get_value(&self) -> &T {
        self.value
            .get()
            .as_ref()
            .expect("UnsafeCell shouldn't return a null pointer")
            .assume_init_ref()
    }
}

#[repr(transparent)]
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
            count: AtomicU8::new(2),
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
    /// both joints still exist. The value is guaranteed to exist as long as the
    /// lock exists, even if the other `Joint` is dropped.
    #[must_use]
    pub fn lock(&mut self) -> Option<JointLock<'_, T>> {
        // Increasing the reference count can always be done with Relaxed– New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        let mut current = self.container().count.load(Ordering::Relaxed);

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

                // Normal state: both `Joints` exist, and the other one may or
                // may not be locked. Increment the count.
                n @ (2 | 3) => match self.container().count.compare_exchange_weak(
                    n,
                    n + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        break Some(JointLock {
                            container: self.container,
                            lifetime: PhantomData,
                        })
                    }
                    Err(n) => n,
                },

                // Semi-broken state: the only way we can observe a 4 here is
                // if other locks were leaked. There's nothing really bad that
                // can happen here outside of promulgating the leak, so we just
                // keep the 4 (since, in the worst case, it was our own sibling
                // that leaked, so there could now once again be 2 locks in
                // existence)
                4 => {
                    break Some(JointLock {
                        container: self.container,
                        lifetime: PhantomData,
                    })
                }

                // It shouldn't ever be possible to observe a larger number
                // than 4, because here in `lock` is the only place increments
                // ever happen.
                n => unsafe { debug_unreachable!("Joint count was {n}") },
            }
        }
    }
}

impl<T: Debug> Debug for Joint<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.container().count.load(Ordering::Relaxed) {
            0 | 1 => write!(f, "Joint(<unpaired>)"),

            // Joints can only be locked if they're mutably borrowed, so we know
            // that we're not locked.
            _ => write!(f, "Joint(<paired>)"),
        }
    }
}

impl<T> Drop for Joint<T> {
    fn drop(&mut self) {
        let count = &self.container().count;

        let mut current = count.load(Ordering::Acquire);

        // Note that all of the failures in the compare-exchanges here are
        // Acquire ordering, even on failure, because failures could indicate
        // that the other handle dropped, meaning that we need to acquire its
        // changes before we start dropping or deallocating anything.
        // Additionally, note that we *usually* don't need to release anything
        // here, because `Joint` isn't itself capable of writing to `value`
        // (only JointLock can do that, and it *does* release on drop.)
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

                n => match count.compare_exchange_weak(
                    n,
                    n - 1,
                    Ordering::Acquire,
                    Ordering::Acquire,
                ) {
                    // All failures, spurious or otherwise, need to be retried.
                    // There's no "fast escape" case (like there are in other
                    // compare-exchange sites) because we always need to ensure
                    // that n - 1 was stored.
                    Err(n) => n,

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
                        loop {
                            match count.compare_exchange_weak(
                                1,
                                0,
                                // Don't need to acquire in this case because we
                                // also did the drop ourselves.
                                Ordering::Release,
                                Ordering::Relaxed,
                            ) {
                                // We stored a zero; the other Joint will be
                                // responsible for deallocating the container
                                Ok(_) => return,

                                // There was already a 0; the other joint
                                // dropped while we were dropping the value.
                                // Deallocate.
                                //
                                // There's no risk of another thread loading
                                // this same 0, because we know the only other
                                // reference in existence is the other Joint. we
                                // stored a 1, so it can never create more
                                // locks; either it will store a 0 (detected
                                // here) or we'll store a 0 that it will load.
                                Err(0) => {
                                    drop(unsafe { Box::from_raw(self.container.as_ptr()) });
                                    return;
                                }

                                // Spurious failure; retry
                                Err(1) => continue,

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
                    }

                    // The other joint is currently locked, so we don't have to
                    // do anything. The other joint's lock will take care of
                    // dropping the data.
                    Ok(3) => return,

                    // Semi-broken state; being here means that a lock leaked at
                    // some point. We can attempt some repair here: we know
                    // that, in the worst case, the other Joint exists and has
                    // at most one lock, so we can set the count to 2. We do
                    // this by' claiming that the compare-exchange failed and
                    // the current count is 3 (which means it'll decrement to
                    // 2).
                    Ok(4) => 3,

                    // It isn't possible for the count to be > 4, because the
                    // only place it ever increments doesn't increase it past 4.
                    Ok(n) => unsafe { debug_unreachable!("Joint count was {n}") },
                },
            }
        }
    }
}

#[repr(transparent)]
pub struct JointLock<'a, T> {
    container: NonNull<JointContainer<T>>,
    lifetime: PhantomData<&'a mut Joint<T>>,
}

// Theoretically we could *not* add `Send` or `Sync` to the lock type; this
// loosen ordering restrictions on its drop implementation, since we could
// guarantee it stayed in the same thread as its parent. However, that would
// preclude its use in certain convenient cases (like in rayon, or across await
// boundaries in Send async functions), so we add them anyway.
unsafe impl<T: Send + Sync> Send for JointLock<'_, T> {}
unsafe impl<T: Send + Sync> Sync for JointLock<'_, T> {}

impl<T> JointLock<'_, T> {
    #[inline]
    #[must_use]
    fn container(&self) -> &JointContainer<T> {
        unsafe { self.container.as_ref() }
    }
}

impl<T> Deref for JointLock<'_, T> {
    type Target = T;

    #[inline]
    #[must_use]
    fn deref(&self) -> &Self::Target {
        // Safety: if a JointLock exists, it's guaranteed that the value will
        // be alive for at least the duration of the lock
        unsafe { self.container().get_value() }
    }
}

impl<T: Debug> Debug for JointLock<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<T> Drop for JointLock<'_, T> {
    fn drop(&mut self) {
        let count = &self.container().count;

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
            // this lock existed. We've already stored the decrement, which
            // means we've taken responsibility for attempting to drop (and that
            // future attempts to lock will now fail)
            2 => {
                unsafe { self.container().drop_value_in_place() }

                // Now that the drop is finished, we can store a 0, so that our
                // parent Joint knows to drop the container itself. There's no
                // need at this point to compare-exchange, since we're
                // guaranteed that the other joint is gone and that our parent
                // joint won't drop before we're done dropping ourselves.
                // We release-store the 0 so that the drop-in-place is visible
                // to our parent.
                count.store(0, Ordering::Release)
            }

            // If the count is higher than two, the value is still alive
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
}
