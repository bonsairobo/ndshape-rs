use crate::Shape;

macro_rules! impl_shape2 {
    ($name:ident, $scalar:ident) => {
        #[derive(Clone)]
        pub struct $name {
            array: [$scalar; 2],
            strides: [$scalar; 2],
            size: $scalar,
        }

        impl $name {
            pub fn new([x, y]: [$scalar; 2]) -> Self {
                Self {
                    array: [x, y],
                    strides: [1, x],
                    size: x * y,
                }
            }
        }

        impl Shape<$scalar, 2> for $name {
            #[inline]
            fn as_array(&self) -> [$scalar; 2] {
                self.array
            }

            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 2]) -> $scalar {
                p[0] + self.strides[1].wrapping_mul(p[1])
            }

            #[inline]
            fn delinearize(&self, i: $scalar) -> [$scalar; 2] {
                let y = i / self.strides[1];
                let x = i % self.strides[1];
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
            array: [$scalar; 3],
            strides: [$scalar; 3],
            size: $scalar,
        }

        impl $name {
            pub fn new([x, y, z]: [$scalar; 3]) -> Self {
                Self {
                    array: [x, y, z],
                    strides: [1, x, x * y],
                    size: x * y * z,
                }
            }
        }

        impl Shape<$scalar, 3> for $name {
            #[inline]
            fn as_array(&self) -> [$scalar; 3] {
                self.array
            }

            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 3]) -> $scalar {
                p[0] + self.strides[1].wrapping_mul(p[1]) + self.strides[2].wrapping_mul(p[2])
            }

            #[inline]
            fn delinearize(&self, mut i: $scalar) -> [$scalar; 3] {
                let z = i / self.strides[2];
                i -= z * self.strides[2];
                let y = i / self.strides[1];
                let x = i % self.strides[1];
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
            array: [$scalar; 4],
            strides: [$scalar; 4],
            size: $scalar,
        }

        impl $name {
            pub fn new([x, y, z, w]: [$scalar; 4]) -> Self {
                Self {
                    array: [x, y, z, w],
                    strides: [1, x, x * y, x * y * z],
                    size: x * y * z * w,
                }
            }
        }

        impl Shape<$scalar, 4> for $name {
            #[inline]
            fn as_array(&self) -> [$scalar; 4] {
                self.array
            }

            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 4]) -> $scalar {
                p[0] + self.strides[1].wrapping_mul(p[1])
                    + self.strides[2].wrapping_mul(p[2])
                    + self.strides[3].wrapping_mul(p[3])
            }

            #[inline]
            fn delinearize(&self, mut i: $scalar) -> [$scalar; 4] {
                let w = i / self.strides[3];
                i -= w * self.strides[3];
                let z = i / self.strides[2];
                i -= z * self.strides[2];
                let y = i / self.strides[1];
                let x = i % self.strides[1];
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
            array: [$scalar; 2],
            shifts: [$scalar; 2],
            masks: [$scalar; 2],
            size: $scalar,
        }

        impl $name {
            pub fn new([x, y]: [$scalar; 2]) -> Self {
                let y_shift = x;
                Self {
                    array: [1 << x, 1 << y],
                    shifts: [0, y_shift],
                    size: 1 << x + y,
                    masks: [!(!0 << x), !(!0 << y) << y_shift],
                }
            }
        }

        impl Shape<$scalar, 2> for $name {
            #[inline]
            fn as_array(&self) -> [$scalar; 2] {
                self.array
            }

            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 2]) -> $scalar {
                (p[1] << self.shifts[1]) | p[0]
            }

            #[inline]
            fn delinearize(&self, i: $scalar) -> [$scalar; 2] {
                [i & self.masks[0], (i & self.masks[1]) >> self.shifts[1]]
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
            array: [$scalar; 3],
            shifts: [$scalar; 3],
            masks: [$scalar; 3],
            size: $scalar,
        }

        impl $name {
            pub fn new([x, y, z]: [$scalar; 3]) -> Self {
                let y_shift = x;
                let z_shift = x + y;
                Self {
                    array: [1 << x, 1 << y, 1 << z],
                    shifts: [0, y_shift, z_shift],
                    masks: [!(!0 << x), !(!0 << y) << y_shift, !(!0 << z) << z_shift],
                    size: 1 << x + y + z,
                }
            }
        }

        impl Shape<$scalar, 3> for $name {
            #[inline]
            fn as_array(&self) -> [$scalar; 3] {
                self.array
            }

            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 3]) -> $scalar {
                (p[2] << self.shifts[2]) | (p[1] << self.shifts[1]) | p[0]
            }

            #[inline]
            fn delinearize(&self, i: $scalar) -> [$scalar; 3] {
                [
                    i & self.masks[0],
                    (i & self.masks[1]) >> self.shifts[1],
                    (i & self.masks[2]) >> self.shifts[2],
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
            array: [$scalar; 4],
            shifts: [$scalar; 4],
            masks: [$scalar; 4],
            size: $scalar,
        }

        impl $name {
            pub fn new([x, y, z, w]: [$scalar; 4]) -> Self {
                let y_shift = x;
                let z_shift = x + y;
                let w_shift = x + y + z;
                Self {
                    array: [1 << x, 1 << y, 1 << z, 1 << w],
                    size: 1 << (x + y + z + w),
                    shifts: [0, y_shift, z_shift, w_shift],
                    masks: [
                        !(!0 << x),
                        !(!0 << y) << y_shift,
                        !(!0 << z) << z_shift,
                        !(!0 << w) << w_shift,
                    ],
                }
            }
        }

        impl Shape<$scalar, 4> for $name {
            #[inline]
            fn as_array(&self) -> [$scalar; 4] {
                self.array
            }

            #[inline]
            fn size(&self) -> $scalar {
                self.size
            }

            #[inline]
            fn linearize(&self, p: [$scalar; 4]) -> $scalar {
                (p[2] << self.shifts[2]) | (p[1] << self.shifts[1]) | p[0]
            }

            #[inline]
            fn delinearize(&self, i: $scalar) -> [$scalar; 4] {
                [
                    i & self.masks[0],
                    (i & self.masks[1]) >> self.shifts[1],
                    (i & self.masks[2]) >> self.shifts[2],
                    (i & self.masks[3]) >> self.shifts[3],
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
