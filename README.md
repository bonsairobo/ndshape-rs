# ndshape

Simple, fast linearization of nonnegative 2D, 3D, and 4D coordinates.

```rust
use ndshape::{Shape, ConstShape3u32, ConstShape4u32, ConstPow2Shape3u32, Shape3u32};

// An arbitrary shape.
let shape = ConstShape3u32::<5, 6, 7>;
let index = shape.linearize([1, 2, 3]);
assert_eq!(index, 101);
assert_eq!(shape.delinearize(index), [1, 2, 3]);

// A shape with power-of-two dimensions
// This allows us to use bit shifting and masking for linearization.
let shape = ConstPow2Shape3u32::<1, 2, 3>; // These are number of bits per dimension.
let index = shape.linearize([1, 2, 3]);
assert_eq!(index, 0b011_10_1);
assert_eq!(shape.delinearize(index), [1, 2, 3]);

// A runtime shape.
let shape = Shape3u32::new([5, 6, 7]);
let index = shape.linearize([1, 2, 3]);
assert_eq!(index, 101);
assert_eq!(shape.delinearize(index), [1, 2, 3]);

// Use a shape for indexing an array in 4D.
// Step X, then Y, then Z, since that results in monotonic increasing indices.
// (Believe it or not, Rust's N-dimensional array indexing is significantly slower than this).
let shape = ConstShape4u32::<5, 6, 7, 8>;
let data = [0; 5 * 6 * 7 * 8];
for w in 0..8 {
    for z in 0..7 {
        for y in 0..6 {
            for x in 0..5 {
                let i = shape.linearize([x, y, z, w]);
                assert_eq!(0, data[i as usize]);
            }
        }
    }
}
```
