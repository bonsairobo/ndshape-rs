//! Simple, fast linearization of nonnegative 2D, 3D, and 4D coordinates.
//!
//! The canonical choice of linearization function is row-major, i.e. stepping linearly through an N dimensional array would
//! step by X first, then Y, then Z, etc, assuming that `[T; N]` coordinates are provided as `[X, Y, Z, ...]`. More explicitly:
//!
//! ```text
//! linearize([x, y, z, ...]) = x + X_SIZE * y + X_SIZE * Y_SIZE * z + ...
//! ```
//!
//! Of course, to achieve a different layout, one only needs to choose a different permutation of coordinates. For example,
//! column-major layout would require coordinates specified as `[..., Z, Y, X]`. For a 3D layout where each Y level set is
//! contiguous in memory, either layout `[X, Z, Y]` or `[Z, X, Y]` would work.
//!
//! # Code Example
//!
//! ```
//! use ndshape::{Shape, ConstShape3u32, ConstShape4u32, ConstPow2Shape3u32, Shape3u32};
//!
//! // An arbitrary shape.
//! let shape = ConstShape3u32::<5, 6, 7>;
//! let index = shape.linearize([1, 2, 3]);
//! assert_eq!(index, 101);
//! assert_eq!(shape.delinearize(index), [1, 2, 3]);
//!
//! // A shape with power-of-two dimensions
//! // This allows us to use bit shifting and masking for linearization.
//! let shape = ConstPow2Shape3u32::<1, 2, 3>; // These are number of bits per dimension.
//! let index = shape.linearize([1, 2, 3]);
//! assert_eq!(index, 0b011_10_1);
//! assert_eq!(shape.delinearize(index), [1, 2, 3]);
//!
//! // A runtime shape.
//! let shape = Shape3u32::new([5, 6, 7]);
//! let index = shape.linearize([1, 2, 3]);
//! assert_eq!(index, 101);
//! assert_eq!(shape.delinearize(index), [1, 2, 3]);
//!
//! // Use a shape for indexing an array in 4D.
//! // Step X, then Y, then Z, since that results in monotonic increasing indices.
//! // (Believe it or not, Rust's N-dimensional array indexing is significantly slower than this).
//! let shape = ConstShape4u32::<5, 6, 7, 8>;
//! let data = [0; 5 * 6 * 7 * 8];
//! for w in 0..8 {
//!     for z in 0..7 {
//!         for y in 0..6 {
//!             for x in 0..5 {
//!                 let i = shape.linearize([x, y, z, w]);
//!                 assert_eq!(0, data[i as usize]);
//!             }
//!         }
//!     }
//! }
//! ```

mod const_shape;
mod shape;

pub use const_shape::*;
pub use shape::*;

/// The shape of an array with unspecified dimensionality.
pub trait AbstractShape<T, V> {
    /// The number of elements in an array with this shape.
    fn size(&self) -> T;
    /// Translates a vector `V` (with an unspecified number of dimensions) into a single number `T` that can be used for
    /// linear indexing.
    fn linearize(&self, p: V) -> T;
    /// The inverse of `linearize`.
    fn delinearize(&self, i: T) -> V;
}

/// The shape of an `N`-dimensional array.
pub trait Shape<T, const N: usize> {
    /// The number of elements in an array with this shape.
    fn size(&self) -> T;
    /// Translate an `N`-dimensional vector into a single number `T` that can be used for linear indexing.
    fn linearize(&self, p: [T; N]) -> T;
    /// The inverse of `linearize`.
    fn delinearize(&self, i: T) -> [T; N];
}

/// A constant shape of an `N`-dimensional array.
pub trait ConstShape<T, const N: usize> {
    /// The number of elements in an array with this shape.
    const SIZE: T;
    /// Translate an `N`-dimensional vector into a single number `T` that can be used for linear indexing.
    fn linearize(p: [T; N]) -> T;
    /// The inverse of `linearize`.
    fn delinearize(i: T) -> [T; N];
}

impl<S, T, const N: usize> AbstractShape<T, [T; N]> for S
where
    S: Shape<T, N>,
{
    #[inline]
    fn size(&self) -> T {
        self.size()
    }
    #[inline]
    fn linearize(&self, p: [T; N]) -> T {
        self.linearize(p)
    }
    #[inline]
    fn delinearize(&self, i: T) -> [T; N] {
        self.delinearize(i)
    }
}

impl<S, T, const N: usize> Shape<T, N> for S
where
    S: ConstShape<T, N>,
{
    #[inline]
    fn size(&self) -> T {
        S::SIZE
    }
    #[inline]
    fn linearize(&self, p: [T; N]) -> T {
        S::linearize(p)
    }
    #[inline]
    fn delinearize(&self, i: T) -> [T; N] {
        S::delinearize(i)
    }
}
