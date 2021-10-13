use crate::Shape;

macro_rules! impl_shape2 {
    ($name:ident, $scalar:ident) => {
        #[derive(Clone)]
        pub struct $name {
            size: $scalar,
            y_stride: $scalar,
        }

        impl $name {
            pub fn new([x, y]: [$scalar; 2]) -> Self {
                Self {
                    size: x * y,
                    y_stride: x,
                }
            }
        }

        impl Shape<$scalar, 2> for $name {
            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 2]) -> $scalar {
                p[0] + self.y_stride * p[1]
            }

            #[inline]
            fn delinearize(&self, i: $scalar) -> [$scalar; 2] {
                let y = i / self.y_stride;
                let x = i % self.y_stride;
                [x, y]
            }
        }
    };
}

impl_shape2!(Shape2u8, u8);
impl_shape2!(Shape2u16, u16);
impl_shape2!(Shape2u32, u32);
impl_shape2!(Shape2u64, u64);

impl_shape2!(Shape2i8, i8);
impl_shape2!(Shape2i16, i16);
impl_shape2!(Shape2i32, i32);
impl_shape2!(Shape2i64, i64);

macro_rules! impl_shape3 {
    ($name:ident, $scalar:ident) => {
        #[derive(Clone)]
        pub struct $name {
            size: $scalar,
            y_stride: $scalar,
            z_stride: $scalar,
        }

        impl $name {
            pub fn new([x, y, z]: [$scalar; 3]) -> Self {
                Self {
                    size: x * y * z,
                    y_stride: x,
                    z_stride: x * y,
                }
            }
        }

        impl Shape<$scalar, 3> for $name {
            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 3]) -> $scalar {
                p[0] + self.y_stride * p[1] + self.z_stride * p[2]
            }

            #[inline]
            fn delinearize(&self, mut i: $scalar) -> [$scalar; 3] {
                let z = i / self.z_stride;
                i -= z * self.z_stride;
                let y = i / self.y_stride;
                let x = i % self.y_stride;
                [x, y, z]
            }
        }
    };
}

impl_shape3!(Shape3u8, u8);
impl_shape3!(Shape3u16, u16);
impl_shape3!(Shape3u32, u32);
impl_shape3!(Shape3u64, u64);

impl_shape3!(Shape3i8, i8);
impl_shape3!(Shape3i16, i16);
impl_shape3!(Shape3i32, i32);
impl_shape3!(Shape3i64, i64);

macro_rules! impl_shape4 {
    ($name:ident, $scalar:ident) => {
        #[derive(Clone)]
        pub struct $name {
            size: $scalar,
            y_stride: $scalar,
            z_stride: $scalar,
            w_stride: $scalar,
        }

        impl $name {
            pub fn new([x, y, z, w]: [$scalar; 4]) -> Self {
                Self {
                    size: x * y * z * w,
                    y_stride: x,
                    z_stride: x * y,
                    w_stride: x * y * z,
                }
            }
        }

        impl Shape<$scalar, 4> for $name {
            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 4]) -> $scalar {
                p[0] + self.y_stride * p[1] + self.z_stride * p[2] + self.w_stride * p[3]
            }

            #[inline]
            fn delinearize(&self, mut i: $scalar) -> [$scalar; 4] {
                let w = i / self.w_stride;
                i -= w * self.w_stride;
                let z = i / self.z_stride;
                i -= z * self.z_stride;
                let y = i / self.y_stride;
                let x = i % self.y_stride;
                [x, y, z, w]
            }
        }
    };
}

impl_shape4!(Shape4u8, u8);
impl_shape4!(Shape4u16, u16);
impl_shape4!(Shape4u32, u32);
impl_shape4!(Shape4u64, u64);

impl_shape4!(Shape4i8, i8);
impl_shape4!(Shape4i16, i16);
impl_shape4!(Shape4i32, i32);
impl_shape4!(Shape4i64, i64);

macro_rules! impl_pow2_shape2 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone)]
        pub struct $name {
            size: $scalar,
            y_shift: $scalar,
            x_mask: $scalar,
            y_mask: $scalar,
        }

        impl $name {
            pub fn new([x, y]: [$scalar; 2]) -> Self {
                let y_shift = x;
                Self {
                    size: 1 << x + y,
                    y_shift,
                    x_mask: !(!0 << x),
                    y_mask: !(!0 << y) << y_shift,
                }
            }
        }

        impl Shape<$scalar, 2> for $name {
            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 2]) -> $scalar {
                (p[1] << self.y_shift) | p[0]
            }

            #[inline]
            fn delinearize(&self, i: $scalar) -> [$scalar; 2] {
                [i & self.x_mask, (i & self.y_mask) >> self.y_shift]
            }
        }
    };
}

impl_pow2_shape2!(Pow2Shape2u8, u8);
impl_pow2_shape2!(Pow2Shape2u16, u16);
impl_pow2_shape2!(Pow2Shape2u32, u32);
impl_pow2_shape2!(Pow2Shape2u64, u64);

impl_pow2_shape2!(Pow2Shape2i8, i8);
impl_pow2_shape2!(Pow2Shape2i16, i16);
impl_pow2_shape2!(Pow2Shape2i32, i32);
impl_pow2_shape2!(Pow2Shape2i64, i64);

macro_rules! impl_pow2_shape3 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone)]
        pub struct $name {
            size: $scalar,
            y_shift: $scalar,
            z_shift: $scalar,
            x_mask: $scalar,
            y_mask: $scalar,
            z_mask: $scalar,
        }

        impl $name {
            pub fn new([x, y, z]: [$scalar; 3]) -> Self {
                let y_shift = x;
                let z_shift = x + y;
                Self {
                    size: 1 << x + y + z,
                    y_shift,
                    z_shift,
                    x_mask: !(!0 << x),
                    y_mask: !(!0 << y) << y_shift,
                    z_mask: !(!0 << z) << z_shift,
                }
            }
        }

        impl Shape<$scalar, 3> for $name {
            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 3]) -> $scalar {
                (p[2] << self.z_shift) | (p[1] << self.y_shift) | p[0]
            }

            #[inline]
            fn delinearize(&self, i: $scalar) -> [$scalar; 3] {
                [
                    i & self.x_mask,
                    (i & self.y_mask) >> self.y_shift,
                    (i & self.z_mask) >> self.z_shift,
                ]
            }
        }
    };
}

impl_pow2_shape3!(Pow2Shape3u8, u8);
impl_pow2_shape3!(Pow2Shape3u16, u16);
impl_pow2_shape3!(Pow2Shape3u32, u32);
impl_pow2_shape3!(Pow2Shape3u64, u64);

impl_pow2_shape3!(Pow2Shape3i8, i8);
impl_pow2_shape3!(Pow2Shape3i16, i16);
impl_pow2_shape3!(Pow2Shape3i32, i32);
impl_pow2_shape3!(Pow2Shape3i64, i64);

macro_rules! impl_pow2_shape4 {
    ($name:ident, $scalar:ty) => {
        #[derive(Clone)]
        pub struct $name {
            size: $scalar,
            y_shift: $scalar,
            z_shift: $scalar,
            w_shift: $scalar,
            x_mask: $scalar,
            y_mask: $scalar,
            z_mask: $scalar,
            w_mask: $scalar,
        }

        impl $name {
            pub fn new([x, y, z, w]: [$scalar; 4]) -> Self {
                let y_shift = x;
                let z_shift = x + y;
                let w_shift = x + y + z;
                Self {
                    size: 1 << (x + y + z + w),
                    y_shift,
                    z_shift,
                    w_shift,
                    x_mask: !(!0 << x),
                    y_mask: !(!0 << y) << y_shift,
                    z_mask: !(!0 << z) << z_shift,
                    w_mask: !(!0 << w) << w_shift,
                }
            }
        }

        impl Shape<$scalar, 4> for $name {
            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 4]) -> $scalar {
                (p[2] << self.z_shift) | (p[1] << self.y_shift) | p[0]
            }

            #[inline]
            fn delinearize(&self, i: $scalar) -> [$scalar; 4] {
                [
                    i & self.x_mask,
                    (i & self.y_mask) >> self.y_shift,
                    (i & self.z_mask) >> self.z_shift,
                    (i & self.w_mask) >> self.w_shift,
                ]
            }
        }
    };
}

impl_pow2_shape4!(Pow2Shape4u8, u8);
impl_pow2_shape4!(Pow2Shape4u16, u16);
impl_pow2_shape4!(Pow2Shape4u32, u32);
impl_pow2_shape4!(Pow2Shape4u64, u64);

impl_pow2_shape4!(Pow2Shape4i8, i8);
impl_pow2_shape4!(Pow2Shape4i16, i16);
impl_pow2_shape4!(Pow2Shape4i32, i32);
impl_pow2_shape4!(Pow2Shape4i64, i64);
