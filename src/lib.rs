//! Simple, fast linearization of 2D, 3D, and 4D coordinates.
//!
//! The canonical choice of linearization function is row-major, i.e. stepping linearly through an N dimensional array would
//! step by X first, then Y, then Z, etc, assuming that `[T; N]` coordinates are provided as `[X, Y, Z, ...]`. More explicitly:
//!
//! ```text
//! linearize([x, y, z, ...]) = x + X_SIZE * y + X_SIZE * Y_SIZE * z + ...
//! ```
//!
//! To achieve a different layout, one only needs to choose a different permutation of coordinates. For example, column-major
//! layout would require coordinates specified as `[..., Z, Y, X]`. For a 3D layout where each Y level set is contiguous in
//! memory, either layout `[X, Z, Y]` or `[Z, X, Y]` would work.
//!
//! # Example: Indexing Multidimensional Arrays
//!
//! ```
//! use ndshape::{Shape, ConstShape3u32, ConstShape4u32, ConstPow2Shape3u32, RuntimeShape};
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
//! let shape = RuntimeShape::<u32, 3>::new([5, 6, 7]);
//! let index = shape.linearize([1, 2, 3]);
//! assert_eq!(index, 101);
//! assert_eq!(shape.delinearize(index), [1, 2, 3]);
//!
//! // Use a shape for indexing an array in 4D.
//! // Step X, then Y, then Z, since that results in monotonic increasing indices.
//! // (Believe it or not, Rust's N-dimensional array (e.g. `[[T; N]; M]`)
//! // indexing is significantly slower than this).
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
//!
//! # Example: Negative Strides with Modular Arithmetic
//!
//! It is often beneficial to linearize a negative vector that results in a negative linear "stride." But when using unsigned
//! linear indices, a negative stride would require a modular arithmetic representation, where e.g. `-1` maps to `u32::MAX`.
//! This works fine with any [`Shape`](crate::Shape). You just need to be sure to use modular arithmetic with the resulting
//! linear strides, e.g. [`u32::wrapping_add`](u32::wrapping_add) and [`u32::wrapping_mul`](u32::wrapping_mul). Also, it is not
//! possible to delinearize a negative stride with modular arithmetic. For that, you must use signed integer coordinates.
//!
//! ```
//! use ndshape::{Shape, ConstShape3u32, ConstShape3i32};
//!
//! let shape = ConstShape3u32::<10, 10, 10>;
//! let stride = shape.linearize([0, -1i32 as u32, 0]);
//! assert_eq!(stride, -10i32 as u32);
//!
//! // Delinearize does not work with unsigned coordinates!
//! assert_ne!(shape.delinearize(stride), [0, -1i32 as u32, 0]);
//! assert_eq!(shape.delinearize(stride), [6, 8, 42949672]);
//!
//! let shape = ConstShape3i32::<10, 10, 10>;
//! let stride = shape.linearize([0, -1, 0]);
//! assert_eq!(stride, -10);
//!
//! // Delinearize works with signed coordinates.
//! assert_eq!(shape.delinearize(stride), [0, -1, 0]);
//! ```

mod const_shape;
mod runtime_shape;

pub use const_shape::*;
pub use runtime_shape::*;

/// The shape of an array with unspecified dimensionality.
pub trait AbstractShape<Coord, Vector> {
    /// The number of elements in an array with this shape.
    fn size(&self) -> Coord;
    /// Translates a vector `V` (with an unspecified number of dimensions) into a single number `T` that can be used for
    /// linear indexing.
    fn linearize(&self, p: Vector) -> Coord;
    /// The inverse of `linearize`.
    fn delinearize(&self, i: Coord) -> Vector;
}

/// The shape of an `N`-dimensional array.
pub trait Shape<const N: usize> {
    type Coord;

    /// The number of elements in an array with this shape.
    fn size(&self) -> Self::Coord;
    /// The same as `self.size() as usize`.
    fn usize(&self) -> usize;
    /// The dimensions of the shape.
    fn as_array(&self) -> [Self::Coord; N];
    /// Translate an `N`-dimensional vector into a single number `T` that can be used for linear indexing.
    fn linearize(&self, p: [Self::Coord; N]) -> Self::Coord;
    /// The inverse of `linearize`.
    fn delinearize(&self, i: Self::Coord) -> [Self::Coord; N];
}

/// A constant shape of an `N`-dimensional array.
pub trait ConstShape<const N: usize> {
    type Coord;

    /// The number of elements in an array with this shape.
    const SIZE: Self::Coord;
    /// Same as `Self::SIZE as usize`.
    const USIZE: usize;
    /// The dimensions of the shape.
    const ARRAY: [Self::Coord; N];
    /// Translate an `N`-dimensional vector into a single number `T` that can be used for linear indexing.
    fn linearize(p: [Self::Coord; N]) -> Self::Coord;
    /// The inverse of `linearize`.
    fn delinearize(i: Self::Coord) -> [Self::Coord; N];
}

impl<S, const N: usize> AbstractShape<S::Coord, [S::Coord; N]> for S
where
    S: Shape<N>,
{
    #[inline]
    fn size(&self) -> S::Coord {
        self.size()
    }
    #[inline]
    fn linearize(&self, p: [S::Coord; N]) -> S::Coord {
        self.linearize(p)
    }
    #[inline]
    fn delinearize(&self, i: S::Coord) -> [S::Coord; N] {
        self.delinearize(i)
    }
}

impl<S, const N: usize> Shape<N> for S
where
    S: ConstShape<N>,
{
    type Coord = S::Coord;

    #[inline]
    fn size(&self) -> Self::Coord {
        S::SIZE
    }
    #[inline]
    fn usize(&self) -> usize {
        S::USIZE
    }
    #[inline]
    fn as_array(&self) -> [Self::Coord; N] {
        S::ARRAY
    }
    #[inline]
    fn linearize(&self, p: [Self::Coord; N]) -> Self::Coord {
        S::linearize(p)
    }
    #[inline]
    fn delinearize(&self, i: Self::Coord) -> [Self::Coord; N] {
        S::delinearize(i)
    }
}
