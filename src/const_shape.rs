use crate::{AbstractShape, ConstShape, Shape};

use static_assertions::assert_impl_all;

macro_rules! impl_const_shape2 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar>;

        impl<const X: $scalar, const Y: $scalar> $name<X, Y> {
            pub const STRIDES: [$scalar; 2] = [1, X];
        }

        impl<const X: $scalar, const Y: $scalar> ConstShape<2> for $name<X, Y> {
            type Coord = $scalar;

            const ARRAY: [$scalar; 2] = [X, Y];
            const SIZE: $scalar = X * Y;
            const USIZE: usize = Self::SIZE as usize;

            #[inline]
            fn linearize(p: [$scalar; 2]) -> $scalar {
                p[0] + Self::STRIDES[1].wrapping_mul(p[1])
            }

            #[inline]
            fn delinearize(i: $scalar) -> [$scalar; 2] {
                let y = i / Self::STRIDES[1];
                let x = i % Self::STRIDES[1];
                [x, y]
            }
        }

        assert_impl_all!($name<1, 1>: AbstractShape<$scalar, [$scalar; 2]>);
        assert_impl_all!($name<1, 1>: Shape<2>);
    };
}

impl_const_shape2!(ConstShape2u8, u8);
impl_const_shape2!(ConstShape2u16, u16);
impl_const_shape2!(ConstShape2u32, u32);
impl_const_shape2!(ConstShape2u64, u64);
impl_const_shape2!(ConstShape2usize, usize);

impl_const_shape2!(ConstShape2i8, i8);
impl_const_shape2!(ConstShape2i16, i16);
impl_const_shape2!(ConstShape2i32, i32);
impl_const_shape2!(ConstShape2i64, i64);

macro_rules! impl_const_shape3 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar, const Z: $scalar>;

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar> $name<X, Y, Z> {
            pub const STRIDES: [$scalar; 3] = [1, X, X * Y];
        }

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar> ConstShape<3>
            for $name<X, Y, Z>
        {
            type Coord = $scalar;

            const ARRAY: [$scalar; 3] = [X, Y, Z];
            const SIZE: $scalar = X * Y * Z;
            const USIZE: usize = Self::SIZE as usize;

            #[inline]
            fn linearize(p: [$scalar; 3]) -> $scalar {
                p[0] + Self::STRIDES[1].wrapping_mul(p[1]) + Self::STRIDES[2].wrapping_mul(p[2])
            }

            #[inline]
            fn delinearize(mut i: $scalar) -> [$scalar; 3] {
                let z = i / Self::STRIDES[2];
                i -= z * Self::STRIDES[2];
                let y = i / Self::STRIDES[1];
                let x = i % Self::STRIDES[1];
                [x, y, z]
            }
        }

        assert_impl_all!($name<1, 1, 1>: AbstractShape<$scalar, [$scalar; 3]>);
        assert_impl_all!($name<1, 1, 1>: Shape<3>);
    };
}

impl_const_shape3!(ConstShape3u8, u8);
impl_const_shape3!(ConstShape3u16, u16);
impl_const_shape3!(ConstShape3u32, u32);
impl_const_shape3!(ConstShape3u64, u64);
impl_const_shape3!(ConstShape3usize, usize);

impl_const_shape3!(ConstShape3i8, i8);
impl_const_shape3!(ConstShape3i16, i16);
impl_const_shape3!(ConstShape3i32, i32);
impl_const_shape3!(ConstShape3i64, i64);

macro_rules! impl_const_shape4 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar, const Z: $scalar, const W: $scalar>;

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar, const W: $scalar>
            $name<X, Y, Z, W>
        {
            pub const STRIDES: [$scalar; 4] = [1, X, X * Y, X * Y * Z];
        }

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar, const W: $scalar>
            ConstShape<4> for $name<X, Y, Z, W>
        {
            type Coord = $scalar;

            const ARRAY: [$scalar; 4] = [X, Y, Z, W];
            const SIZE: $scalar = X * Y * Z * W;
            const USIZE: usize = Self::SIZE as usize;

            #[inline]
            fn linearize(p: [$scalar; 4]) -> $scalar {
                p[0] +
                    Self::STRIDES[1].wrapping_mul(p[1]) +
                    Self::STRIDES[2].wrapping_mul(p[2]) +
                    Self::STRIDES[3].wrapping_mul(p[3])
            }

            #[inline]
            fn delinearize(mut i: $scalar) -> [$scalar; 4] {
                let w = i / Self::STRIDES[3];
                i -= w * Self::STRIDES[3];
                let z = i / Self::STRIDES[2];
                i -= z * Self::STRIDES[2];
                let y = i / Self::STRIDES[1];
                let x = i % Self::STRIDES[1];
                [x, y, z, w]
            }
        }

        assert_impl_all!($name<1, 1, 1, 1>: AbstractShape<$scalar, [$scalar; 4]>);
        assert_impl_all!($name<1, 1, 1, 1>: Shape<4>);
    };
}

impl_const_shape4!(ConstShape4u8, u8);
impl_const_shape4!(ConstShape4u16, u16);
impl_const_shape4!(ConstShape4u32, u32);
impl_const_shape4!(ConstShape4u64, u64);
impl_const_shape4!(ConstShape4usize, usize);

impl_const_shape4!(ConstShape4i8, i8);
impl_const_shape4!(ConstShape4i16, i16);
impl_const_shape4!(ConstShape4i32, i32);
impl_const_shape4!(ConstShape4i64, i64);

macro_rules! impl_const_pow2_shape2 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar>;

        impl<const X: $scalar, const Y: $scalar> $name<X, Y> {
            pub const SHIFTS: [$scalar; 2] = [0, X];

            pub const MASKS: [$scalar; 2] = [
                !(!0 << X),
                !(!0 << Y) << Self::SHIFTS[1]
            ];
        }

        impl<const X: $scalar, const Y: $scalar> ConstShape<2> for $name<X, Y> {
            type Coord = $scalar;

            const ARRAY: [$scalar; 2] = [1 << X, 1 << Y];
            const SIZE: $scalar = 1 << (X + Y);
            const USIZE: usize = Self::SIZE as usize;

            #[inline]
            fn linearize(p: [$scalar; 2]) -> $scalar {
                (p[1] << Self::SHIFTS[1]) | p[0]
            }

            #[inline]
            fn delinearize(i: $scalar) -> [$scalar; 2] {
                [(i & Self::MASKS[0]), ((i & Self::MASKS[1]) >> Self::SHIFTS[1])]
            }
        }

        assert_impl_all!($name<1, 1>: AbstractShape<$scalar, [$scalar; 2]>);
        assert_impl_all!($name<1, 1>: Shape<2>);
    };
}

impl_const_pow2_shape2!(ConstPow2Shape2u8, u8);
impl_const_pow2_shape2!(ConstPow2Shape2u16, u16);
impl_const_pow2_shape2!(ConstPow2Shape2u32, u32);
impl_const_pow2_shape2!(ConstPow2Shape2u64, u64);
impl_const_pow2_shape2!(ConstPow2Shape2usize, usize);

impl_const_pow2_shape2!(ConstPow2Shape2i8, i8);
impl_const_pow2_shape2!(ConstPow2Shape2i16, i16);
impl_const_pow2_shape2!(ConstPow2Shape2i32, i32);
impl_const_pow2_shape2!(ConstPow2Shape2i64, i64);

macro_rules! impl_const_pow2_shape3 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar, const Z: $scalar>;

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar> $name<X, Y, Z> {
            pub const SHIFTS: [$scalar; 3] = [0, X, X + Y];

            pub const MASKS: [$scalar; 3] = [
                !(!0 << X),
                !(!0 << Y) << Self::SHIFTS[1],
                !(!0 << Z) << Self::SHIFTS[2],
            ];
        }

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar> ConstShape<3>
            for $name<X, Y, Z>
        {
            type Coord = $scalar;

            const ARRAY: [$scalar; 3] = [1 << X, 1 << Y, 1 << Z];
            const SIZE: $scalar = 1 << (X + Y + Z);
            const USIZE: usize = Self::SIZE as usize;

            #[inline]
            fn linearize(p: [$scalar; 3]) -> $scalar {
                (p[2] << Self::SHIFTS[2]) | (p[1] << Self::SHIFTS[1]) | p[0]
            }

            #[inline]
            fn delinearize(i: $scalar) -> [$scalar; 3] {
                [
                    (i & Self::MASKS[0]),
                    ((i & Self::MASKS[1]) >> Self::SHIFTS[1]),
                    ((i & Self::MASKS[2]) >> Self::SHIFTS[2]),
                ]
            }
        }

        assert_impl_all!($name<1, 1, 1>: AbstractShape<$scalar, [$scalar; 3]>);
        assert_impl_all!($name<1, 1, 1>: Shape<3>);
    };
}

impl_const_pow2_shape3!(ConstPow2Shape3u8, u8);
impl_const_pow2_shape3!(ConstPow2Shape3u16, u16);
impl_const_pow2_shape3!(ConstPow2Shape3u32, u32);
impl_const_pow2_shape3!(ConstPow2Shape3u64, u64);
impl_const_pow2_shape3!(ConstPow2Shape3usize, usize);

impl_const_pow2_shape3!(ConstPow2Shape3i8, i8);
impl_const_pow2_shape3!(ConstPow2Shape3i16, i16);
impl_const_pow2_shape3!(ConstPow2Shape3i32, i32);
impl_const_pow2_shape3!(ConstPow2Shape3i64, i64);

macro_rules! impl_const_pow2_shape4 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar, const Z: $scalar, const W: $scalar>;

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar, const W: $scalar>
            $name<X, Y, Z, W>
        {
            pub const SHIFTS: [$scalar; 4] = [0, X, X + Y, X + Y + Z];

            pub const MASKS: [$scalar; 4] = [
                !(!0 << X),
                !(!0 << Y) << Self::SHIFTS[1],
                !(!0 << Z) << Self::SHIFTS[2],
                !(!0 << W) << Self::SHIFTS[3],
            ];
        }

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar, const W: $scalar>
            ConstShape<4> for $name<X, Y, Z, W>
        {
            type Coord = $scalar;

            const ARRAY: [$scalar; 4] = [1 << X, 1 << Y, 1 << Z, 1 << W];
            const SIZE: $scalar = 1 << (X + Y + Z + W);
            const USIZE: usize = Self::SIZE as usize;

            #[inline]
            fn linearize(p: [$scalar; 4]) -> $scalar {
                (p[3] << Self::SHIFTS[3]) | (p[2] << Self::SHIFTS[2]) | (p[1] << Self::SHIFTS[1]) | p[0]
            }

            #[inline]
            fn delinearize(i: $scalar) -> [$scalar; 4] {
                [
                    (i & Self::MASKS[0]),
                    ((i & Self::MASKS[1]) >> Self::SHIFTS[1]),
                    ((i & Self::MASKS[2]) >> Self::SHIFTS[2]),
                    ((i & Self::MASKS[3]) >> Self::SHIFTS[3]),
                ]
            }
        }

        assert_impl_all!($name<1, 1, 1, 1>: AbstractShape<$scalar, [$scalar; 4]>);
        assert_impl_all!($name<1, 1, 1, 1>: Shape<4>);
    };
}

impl_const_pow2_shape4!(ConstPow2Shape4u8, u8);
impl_const_pow2_shape4!(ConstPow2Shape4u16, u16);
impl_const_pow2_shape4!(ConstPow2Shape4u32, u32);
impl_const_pow2_shape4!(ConstPow2Shape4u64, u64);
impl_const_pow2_shape4!(ConstPow2Shape4usize, usize);

impl_const_pow2_shape4!(ConstPow2Shape4i8, i8);
impl_const_pow2_shape4!(ConstPow2Shape4i16, i16);
impl_const_pow2_shape4!(ConstPow2Shape4i32, i32);
impl_const_pow2_shape4!(ConstPow2Shape4i64, i64);
