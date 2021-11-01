use crate::Shape;

#[derive(Clone)]
pub struct RuntimeShape<C, const N: usize> {
    array: [C; N],
    strides: [C; N],
    size: C,
}

macro_rules! impl_shape2 {
    ($scalar:ident) => {
        impl RuntimeShape<$scalar, 2> {
            pub fn new([x, y]: [$scalar; 2]) -> Self {
                Self {
                    array: [x, y],
                    strides: [1, x],
                    size: x * y,
                }
            }
        }

        impl Shape<$scalar, 2> for RuntimeShape<$scalar, 2> {
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

impl_shape2!(u8);
impl_shape2!(u16);
impl_shape2!(u32);
impl_shape2!(u64);

impl_shape2!(i8);
impl_shape2!(i16);
impl_shape2!(i32);
impl_shape2!(i64);

macro_rules! impl_shape3 {
    ($scalar:ident) => {
        impl RuntimeShape<$scalar, 3> {
            pub fn new([x, y, z]: [$scalar; 3]) -> Self {
                Self {
                    array: [x, y, z],
                    strides: [1, x, x * y],
                    size: x * y * z,
                }
            }
        }

        impl Shape<$scalar, 3> for RuntimeShape<$scalar, 3> {
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

impl_shape3!(u8);
impl_shape3!(u16);
impl_shape3!(u32);
impl_shape3!(u64);

impl_shape3!(i8);
impl_shape3!(i16);
impl_shape3!(i32);
impl_shape3!(i64);

macro_rules! impl_shape4 {
    ($scalar:ident) => {
        impl RuntimeShape<$scalar, 4> {
            pub fn new([x, y, z, w]: [$scalar; 4]) -> Self {
                Self {
                    array: [x, y, z, w],
                    strides: [1, x, x * y, x * y * z],
                    size: x * y * z * w,
                }
            }
        }

        impl Shape<$scalar, 4> for RuntimeShape<$scalar, 4> {
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

impl_shape4!(u8);
impl_shape4!(u16);
impl_shape4!(u32);
impl_shape4!(u64);

impl_shape4!(i8);
impl_shape4!(i16);
impl_shape4!(i32);
impl_shape4!(i64);

#[derive(Clone)]
pub struct RuntimePow2Shape<C, const N: usize> {
    array: [C; N],
    shifts: [C; N],
    masks: [C; N],
    size: C,
}

macro_rules! impl_pow2_shape2 {
    ($scalar:ty) => {
        impl RuntimePow2Shape<$scalar, 2> {
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

        impl Shape<$scalar, 2> for RuntimePow2Shape<$scalar, 2> {
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

impl_pow2_shape2!(u8);
impl_pow2_shape2!(u16);
impl_pow2_shape2!(u32);
impl_pow2_shape2!(u64);

impl_pow2_shape2!(i8);
impl_pow2_shape2!(i16);
impl_pow2_shape2!(i32);
impl_pow2_shape2!(i64);

macro_rules! impl_pow2_shape3 {
    ($scalar:ty) => {
        impl RuntimePow2Shape<$scalar, 3> {
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

        impl Shape<$scalar, 3> for RuntimePow2Shape<$scalar, 3> {
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

impl_pow2_shape3!(u8);
impl_pow2_shape3!(u16);
impl_pow2_shape3!(u32);
impl_pow2_shape3!(u64);

impl_pow2_shape3!(i8);
impl_pow2_shape3!(i16);
impl_pow2_shape3!(i32);
impl_pow2_shape3!(i64);

macro_rules! impl_pow2_shape4 {
    ($scalar:ty) => {
        impl RuntimePow2Shape<$scalar, 4> {
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

        impl Shape<$scalar, 4> for RuntimePow2Shape<$scalar, 4> {
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

impl_pow2_shape4!(u8);
impl_pow2_shape4!(u16);
impl_pow2_shape4!(u32);
impl_pow2_shape4!(u64);

impl_pow2_shape4!(i8);
impl_pow2_shape4!(i16);
impl_pow2_shape4!(i32);
impl_pow2_shape4!(i64);
