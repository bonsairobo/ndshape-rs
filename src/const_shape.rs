use crate::{AbstractShape, ConstShape, Shape};

use static_assertions::assert_impl_all;

macro_rules! impl_const_shape2 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar>;

        impl<const X: $scalar, const Y: $scalar> $name<X, Y> {
            pub const Y_STRIDE: $scalar = X;
        }

        impl<const X: $scalar, const Y: $scalar> ConstShape<$scalar, 2> for $name<X, Y> {
            const SIZE: $scalar = X * Y;

            #[inline]
            fn linearize(p: [$scalar; 2]) -> $scalar {
                p[0] + Self::Y_STRIDE * p[1]
            }

            #[inline]
            fn delinearize(i: $scalar) -> [$scalar; 2] {
                let y = i / Self::Y_STRIDE;
                let x = i % Self::Y_STRIDE;
                [x, y]
            }
        }

        assert_impl_all!($name<1, 1>: AbstractShape<$scalar, [$scalar; 2]>);
        assert_impl_all!($name<1, 1>: Shape<$scalar, 2>);
    };
}

impl_const_shape2!(ConstShape2u8, u8);
impl_const_shape2!(ConstShape2u16, u16);
impl_const_shape2!(ConstShape2u32, u32);
impl_const_shape2!(ConstShape2u64, u64);

impl_const_shape2!(ConstShape2i8, i8);
impl_const_shape2!(ConstShape2i16, i16);
impl_const_shape2!(ConstShape2i32, i32);
impl_const_shape2!(ConstShape2i64, i64);

macro_rules! impl_const_shape3 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar, const Z: $scalar>;

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar> $name<X, Y, Z> {
            pub const Y_STRIDE: $scalar = X;
            pub const Z_STRIDE: $scalar = X * Y;
        }

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar> ConstShape<$scalar, 3>
            for $name<X, Y, Z>
        {
            const SIZE: $scalar = X * Y * Z;

            #[inline]
            fn linearize(p: [$scalar; 3]) -> $scalar {
                p[0] + Self::Y_STRIDE * p[1] + Self::Z_STRIDE * p[2]
            }

            #[inline]
            fn delinearize(mut i: $scalar) -> [$scalar; 3] {
                let z = i / Self::Z_STRIDE;
                i -= z * Self::Z_STRIDE;
                let y = i / Self::Y_STRIDE;
                let x = i % Self::Y_STRIDE;
                [x, y, z]
            }
        }

        assert_impl_all!($name<1, 1, 1>: AbstractShape<$scalar, [$scalar; 3]>);
        assert_impl_all!($name<1, 1, 1>: Shape<$scalar, 3>);
    };
}

impl_const_shape3!(ConstShape3u8, u8);
impl_const_shape3!(ConstShape3u16, u16);
impl_const_shape3!(ConstShape3u32, u32);
impl_const_shape3!(ConstShape3u64, u64);

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
            pub const Y_STRIDE: $scalar = X;
            pub const Z_STRIDE: $scalar = X * Y;
            pub const W_STRIDE: $scalar = X * Y * Z;
        }

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar, const W: $scalar>
            ConstShape<$scalar, 4> for $name<X, Y, Z, W>
        {
            const SIZE: $scalar = X * Y * Z * W;

            #[inline]
            fn linearize(p: [$scalar; 4]) -> $scalar {
                p[0] + Self::Y_STRIDE * p[1] + Self::Z_STRIDE * p[2] + Self::W_STRIDE * p[3]
            }

            #[inline]
            fn delinearize(mut i: $scalar) -> [$scalar; 4] {
                let w = i / Self::W_STRIDE;
                i -= w * Self::W_STRIDE;
                let z = i / Self::Z_STRIDE;
                i -= z * Self::Z_STRIDE;
                let y = i / Self::Y_STRIDE;
                let x = i % Self::Y_STRIDE;
                [x, y, z, w]
            }
        }

        assert_impl_all!($name<1, 1, 1, 1>: AbstractShape<$scalar, [$scalar; 4]>);
        assert_impl_all!($name<1, 1, 1, 1>: Shape<$scalar, 4>);
    };
}

impl_const_shape4!(ConstShape4u8, u8);
impl_const_shape4!(ConstShape4u16, u16);
impl_const_shape4!(ConstShape4u32, u32);
impl_const_shape4!(ConstShape4u64, u64);

impl_const_shape4!(ConstShape4i8, i8);
impl_const_shape4!(ConstShape4i16, i16);
impl_const_shape4!(ConstShape4i32, i32);
impl_const_shape4!(ConstShape4i64, i64);

macro_rules! impl_const_pow2_shape2 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar>;

        impl<const X: $scalar, const Y: $scalar> $name<X, Y> {
            pub const Y_SHIFT: $scalar = X;

            pub const X_MASK: $scalar = !(!0 << X);
            pub const Y_MASK: $scalar = !(!0 << Y) << Self::Y_SHIFT;
        }

        impl<const X: $scalar, const Y: $scalar> ConstShape<$scalar, 2> for $name<X, Y> {
            const SIZE: $scalar = 1 << (X + Y);

            #[inline]
            fn linearize(p: [$scalar; 2]) -> $scalar {
                (p[1] << Self::Y_SHIFT) | p[0]
            }

            #[inline]
            fn delinearize(i: $scalar) -> [$scalar; 2] {
                [(i & Self::X_MASK), ((i & Self::Y_MASK) >> Self::Y_SHIFT)]
            }
        }

        assert_impl_all!($name<1, 1>: AbstractShape<$scalar, [$scalar; 2]>);
        assert_impl_all!($name<1, 1>: Shape<$scalar, 2>);
    };
}

impl_const_pow2_shape2!(ConstPow2Shape2u8, u8);
impl_const_pow2_shape2!(ConstPow2Shape2u16, u16);
impl_const_pow2_shape2!(ConstPow2Shape2u32, u32);
impl_const_pow2_shape2!(ConstPow2Shape2u64, u64);

impl_const_pow2_shape2!(ConstPow2Shape2i8, i8);
impl_const_pow2_shape2!(ConstPow2Shape2i16, i16);
impl_const_pow2_shape2!(ConstPow2Shape2i32, i32);
impl_const_pow2_shape2!(ConstPow2Shape2i64, i64);

macro_rules! impl_const_pow2_shape3 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone, Debug, Copy, Eq, PartialEq)]
        pub struct $name<const X: $scalar, const Y: $scalar, const Z: $scalar>;

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar> $name<X, Y, Z> {
            pub const Y_SHIFT: $scalar = X;
            pub const Z_SHIFT: $scalar = X + Y;

            pub const X_MASK: $scalar = !(!0 << X);
            pub const Y_MASK: $scalar = !(!0 << Y) << Self::Y_SHIFT;
            pub const Z_MASK: $scalar = !(!0 << Z) << Self::Z_SHIFT;
        }

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar> ConstShape<$scalar, 3>
            for $name<X, Y, Z>
        {
            const SIZE: $scalar = 1 << (X + Y + Z);

            #[inline]
            fn linearize(p: [$scalar; 3]) -> $scalar {
                (p[2] << Self::Z_SHIFT) | (p[1] << Self::Y_SHIFT) | p[0]
            }

            #[inline]
            fn delinearize(i: $scalar) -> [$scalar; 3] {
                [
                    (i & Self::X_MASK),
                    ((i & Self::Y_MASK) >> Self::Y_SHIFT),
                    ((i & Self::Z_MASK) >> Self::Z_SHIFT),
                ]
            }
        }

        assert_impl_all!($name<1, 1, 1>: AbstractShape<$scalar, [$scalar; 3]>);
        assert_impl_all!($name<1, 1, 1>: Shape<$scalar, 3>);
    };
}

impl_const_pow2_shape3!(ConstPow2Shape3u8, u8);
impl_const_pow2_shape3!(ConstPow2Shape3u16, u16);
impl_const_pow2_shape3!(ConstPow2Shape3u32, u32);
impl_const_pow2_shape3!(ConstPow2Shape3u64, u64);

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
            pub const Y_SHIFT: $scalar = X;
            pub const Z_SHIFT: $scalar = X + Y;
            pub const W_SHIFT: $scalar = X + Y + Z;

            pub const X_MASK: $scalar = !(!0 << X);
            pub const Y_MASK: $scalar = !(!0 << Y) << Self::Y_SHIFT;
            pub const Z_MASK: $scalar = !(!0 << Z) << Self::Z_SHIFT;
            pub const W_MASK: $scalar = !(!0 << Z) << Self::W_SHIFT;
        }

        impl<const X: $scalar, const Y: $scalar, const Z: $scalar, const W: $scalar>
            ConstShape<$scalar, 4> for $name<X, Y, Z, W>
        {
            const SIZE: $scalar = 1 << (X + Y + Z + W);

            #[inline]
            fn linearize(p: [$scalar; 4]) -> $scalar {
                (p[3] << Self::W_SHIFT) | (p[2] << Self::Z_SHIFT) | (p[1] << Self::Y_SHIFT) | p[0]
            }

            #[inline]
            fn delinearize(i: $scalar) -> [$scalar; 4] {
                [
                    (i & Self::X_MASK),
                    ((i & Self::Y_MASK) >> Self::Y_SHIFT),
                    ((i & Self::Z_MASK) >> Self::Z_SHIFT),
                    ((i & Self::W_MASK) >> Self::W_SHIFT),
                ]
            }
        }

        assert_impl_all!($name<1, 1, 1, 1>: AbstractShape<$scalar, [$scalar; 4]>);
        assert_impl_all!($name<1, 1, 1, 1>: Shape<$scalar, 4>);
    };
}

impl_const_pow2_shape4!(ConstPow2Shape4u8, u8);
impl_const_pow2_shape4!(ConstPow2Shape4u16, u16);
impl_const_pow2_shape4!(ConstPow2Shape4u32, u32);
impl_const_pow2_shape4!(ConstPow2Shape4u64, u64);

impl_const_pow2_shape4!(ConstPow2Shape4i8, i8);
impl_const_pow2_shape4!(ConstPow2Shape4i16, i16);
impl_const_pow2_shape4!(ConstPow2Shape4i32, i32);
impl_const_pow2_shape4!(ConstPow2Shape4i64, i64);
