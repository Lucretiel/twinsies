# twinsies

Twinsies is a special shared pointer, similar to an [`Arc`], where two specific
objects (called [`Joint`]) share joint ownership of the underlying object. The
key difference compared to an [`Arc`] is that the underlying object is dropped
when _either_ of the [`Joint`] objects go out of scope.

Because a single [`Joint`] cannot, by itself, keep the shared object alive, it
cannot be dereferenced directly like an [`Arc`]. Instead, it must be locked
with [`.lock()`]. While locked, the object is guaranteed to stay alive as long
as the [`JointLock`] is alive. If the a [`Joint`] is dropped while its partner
is locked, the object stays alive, but it dropped immediately as soon as the
other [`Joint`] is no longer locked.

Twinsies is intended to be used for things like unbuffered channels, join
handles, and async [`Waker`]- cases where some piece of shared state should only
be preserved as long as _both_ halves are still interested in it.

## Example

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

[`arc`]: std::sync::Arc
[`weak`]: std::sync::Weak
[`waker`]: std::task::Waker
[`.lock()`]: Joint::lock
