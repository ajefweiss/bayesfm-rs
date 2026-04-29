use nalgebra::RealField;

/// Evaluate a sum of spherical harmonics for a set of coefficients.
///
/// The maximum order l is determined by the length of the coefficients.
/// All coefficients are assumed to be zero beyond the maximum order `coeffs.len().isqrt() - 1`.
pub fn sph_evaluate<T>(coeffs: &[T], theta: T, phi: T) -> T
where
    T: RealField,
{
    let l_max = { coeffs.len().isqrt() - 1 };

    let mut sum = T::zero();

    for i in 0..l_max {
        for j in (-(i as isize))..=(i as isize) {
            let index = sph_get_index(i, j).unwrap();
            sum += coeffs[index].clone() * sph_ylm(i, j)(theta.clone(), phi.clone());
        }
    }

    sum
}

/// Converts a spherical harmonic index to its (l, m) order and degree.
///
/// Spherical harmonics are indexed in order: (0,0), (1,-1), (1,0), (1,1), (2,-2), ...
/// For a given index, this function returns the corresponding (l, m) pair where:
/// - l is the degree (0, 1, 2, ...)
/// - m is the order (-l to +l)
pub const fn sph_get_lm(index: usize) -> (usize, isize) {
    // For degree l, the starting index is l^2
    // Find l such that l^2 <= index < (l+1)^2
    let l = index.isqrt();

    // Position within the l group (0 to 2l)
    let pos_in_l = index - l * l;

    // m ranges from -l to +l, so m = pos_in_l - l
    let m = (pos_in_l as isize) - (l as isize);

    (l, m)
}

/// Converts (l, m) order and degree to a spherical harmonic index.
///
/// This is the inverse of `sph_get_lm`. Given a pair (l, m), returns `Some(index)` if valid,
/// or `None` if m is outside the valid range [-l, l].
pub const fn sph_get_index(l: usize, m: isize) -> Option<usize> {
    let l_i = l as isize;

    if m < -l_i || m > l_i {
        return None;
    }

    // Index = l^2 + (m + l)
    Some(l * l + ((m + l_i) as usize))
}

#[macro_export]
/// Generates the name of a spherical harmonic function.
macro_rules! sph_name {
    ($prefix: literal, $l: expr, $m: expr) => {
        if $m == 0 {
            concat!($prefix, "_y", stringify!($l), stringify!($m))
        } else if $m > 0 {
            concat!($prefix, "_y", stringify!($l), "+", stringify!($m))
        } else {
            concat!($prefix, "_y", stringify!($l), stringify!($m))
        }
    };
}

pub use sph_name;

/// Returns the real spherical harmonic function of order (l, m), normalized as in geodesy.
pub const fn sph_ylm<T>(l: usize, m: isize) -> impl Fn(T, T) -> T
where
    T: RealField,
{
    match (l, m) {
        (0, 0) => sph_basis::sph_y00,

        (1, -1) => sph_basis::sph_y11s,
        (1, 0) => sph_basis::sph_y10,
        (1, 1) => sph_basis::sph_y11c,

        (2, -2) => sph_basis::sph_y22s,
        (2, -1) => sph_basis::sph_y21s,
        (2, 0) => sph_basis::sph_y20,
        (2, 1) => sph_basis::sph_y21c,
        (2, 2) => sph_basis::sph_y22c,

        (3, -3) => sph_basis::sph_y33s,
        (3, -2) => sph_basis::sph_y32s,
        (3, -1) => sph_basis::sph_y31s,
        (3, 0) => sph_basis::sph_y30,
        (3, 1) => sph_basis::sph_y31c,
        (3, 2) => sph_basis::sph_y32c,
        (3, 3) => sph_basis::sph_y33c,

        (4, -4) => sph_basis::sph_y44s,
        (4, -3) => sph_basis::sph_y43s,
        (4, -2) => sph_basis::sph_y42s,
        (4, -1) => sph_basis::sph_y41s,
        (4, 0) => sph_basis::sph_y40,
        (4, 1) => sph_basis::sph_y41c,
        (4, 2) => sph_basis::sph_y42c,
        (4, 3) => sph_basis::sph_y43c,
        (4, 4) => sph_basis::sph_y44c,

        (5, -5) => sph_basis::sph_y55s,
        (5, -4) => sph_basis::sph_y54s,
        (5, -3) => sph_basis::sph_y53s,
        (5, -2) => sph_basis::sph_y52s,
        (5, -1) => sph_basis::sph_y51s,
        (5, 0) => sph_basis::sph_y50,
        (5, 1) => sph_basis::sph_y51c,
        (5, 2) => sph_basis::sph_y52c,
        (5, 3) => sph_basis::sph_y53c,
        (5, 4) => sph_basis::sph_y54c,
        (5, 5) => sph_basis::sph_y55c,

        (6, -6) => sph_basis::sph_y66s,
        (6, -5) => sph_basis::sph_y65s,
        (6, -4) => sph_basis::sph_y64s,
        (6, -3) => sph_basis::sph_y63s,
        (6, -2) => sph_basis::sph_y62s,
        (6, -1) => sph_basis::sph_y61s,
        (6, 0) => sph_basis::sph_y60,
        (6, 1) => sph_basis::sph_y61c,
        (6, 2) => sph_basis::sph_y62c,
        (6, 3) => sph_basis::sph_y63c,
        (6, 4) => sph_basis::sph_y64c,
        (6, 5) => sph_basis::sph_y65c,
        (6, 6) => sph_basis::sph_y66c,

        (7, -7) => sph_basis::sph_y77s,
        (7, -6) => sph_basis::sph_y76s,
        (7, -5) => sph_basis::sph_y75s,
        (7, -4) => sph_basis::sph_y74s,
        (7, -3) => sph_basis::sph_y73s,
        (7, -2) => sph_basis::sph_y72s,
        (7, -1) => sph_basis::sph_y71s,
        (7, 0) => sph_basis::sph_y70,
        (7, 1) => sph_basis::sph_y71c,
        (7, 2) => sph_basis::sph_y72c,
        (7, 3) => sph_basis::sph_y73c,
        (7, 4) => sph_basis::sph_y74c,
        (7, 5) => sph_basis::sph_y75c,
        (7, 6) => sph_basis::sph_y76c,
        (7, 7) => sph_basis::sph_y77c,

        (8, -8) => sph_basis::sph_y88s,
        (8, -7) => sph_basis::sph_y87s,
        (8, -6) => sph_basis::sph_y86s,
        (8, -5) => sph_basis::sph_y85s,
        (8, -4) => sph_basis::sph_y84s,
        (8, -3) => sph_basis::sph_y83s,
        (8, -2) => sph_basis::sph_y82s,
        (8, -1) => sph_basis::sph_y81s,
        (8, 0) => sph_basis::sph_y80,
        (8, 1) => sph_basis::sph_y81c,
        (8, 2) => sph_basis::sph_y82c,
        (8, 3) => sph_basis::sph_y83c,
        (8, 4) => sph_basis::sph_y84c,
        (8, 5) => sph_basis::sph_y85c,
        (8, 6) => sph_basis::sph_y86c,
        (8, 7) => sph_basis::sph_y87c,
        (8, 8) => sph_basis::sph_y88c,

        _ => unimplemented!(),
    }
}

mod sph_basis {
    use crate::tval;
    use nalgebra::RealField;

    /// Real spherical harmonic Y_0^0(θ, φ).
    pub fn sph_y00<T>(_theta: T, _varphi: T) -> T
    where
        T: RealField,
    {
        T::one()
    }

    /// Real spherical harmonic Y_1^{-1}(θ, φ).
    pub fn sph_y11s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3, usize)).sqrt() * theta.sin() * varphi.sin()
    }

    /// Real spherical harmonic Y_1^0(θ, φ).
    pub fn sph_y10<T>(theta: T, _varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3, usize)).sqrt() * theta.cos()
    }

    /// Real spherical harmonic Y_1^1(θ, φ).
    pub fn sph_y11c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3, usize)).sqrt() * theta.sin() * varphi.cos()
    }

    /// Real spherical harmonic Y_2^{-2}(θ, φ).
    pub fn sph_y22s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(15, usize)).sqrt() / tval!(2, usize)
            * theta.sin().powi(2)
            * (tval!(2, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_2^{-1}(θ, φ).
    pub fn sph_y21s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(15, usize)).sqrt() / tval!(2, usize) * (tval!(2, usize) * theta).sin() * varphi.sin()
    }

    /// Real spherical harmonic Y_2^0(θ, φ).
    pub fn sph_y20<T>(theta: T, _varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(5, usize)).sqrt() / tval!(2, usize)
            * (tval!(3, usize) * theta.cos().powi(2) - T::one())
    }

    /// Real spherical harmonic Y_2^1(θ, φ).
    pub fn sph_y21c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(15, usize)).sqrt() / tval!(2, usize) * (tval!(2, usize) * theta).sin() * varphi.cos()
    }

    /// Real spherical harmonic Y_2^2(θ, φ).
    pub fn sph_y22c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(15, usize)).sqrt() / tval!(2, usize)
            * theta.sin().powi(2)
            * (tval!(2, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_3^{-3}(θ, φ).
    pub fn sph_y33s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(35, usize) / tval!(8, usize)).sqrt()
            * theta.sin().powi(3)
            * (tval!(3, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_3^{-2}(θ, φ).
    pub fn sph_y32s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(105, usize) / tval!(4, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * theta.cos()
            * (tval!(2, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_3^{-1}(θ, φ).
    pub fn sph_y31s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(21, usize) / tval!(8, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(5, usize) * theta.cos().powi(2) - T::one())
            * varphi.sin()
    }

    /// Real spherical harmonic Y_3^0(θ, φ).
    pub fn sph_y30<T>(theta: T, _varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(7, usize)).sqrt() / tval!(2, usize)
            * (tval!(5, usize) * theta.clone().cos().powi(3) - tval!(3, usize) * theta.cos())
    }

    /// Real spherical harmonic Y_3^1(θ, φ).
    pub fn sph_y31c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(21, usize) / tval!(8, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(5, usize) * theta.cos().powi(2) - T::one())
            * varphi.cos()
    }

    /// Real spherical harmonic Y_3^2(θ, φ).
    pub fn sph_y32c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(105, usize) / tval!(4, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * theta.cos()
            * (tval!(2, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_3^3(θ, φ).
    pub fn sph_y33c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(35, usize) / tval!(8, usize)).sqrt()
            * theta.sin().powi(3)
            * (tval!(3, usize) * varphi).cos()
    }

    // l=4 functions

    /// Real spherical harmonic Y_4^{-4}(θ, φ).
    pub fn sph_y44s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(315, usize) / tval!(64, usize)).sqrt()
            * theta.sin().powi(4)
            * (tval!(4, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_4^{-3}(θ, φ).
    pub fn sph_y43s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(315, usize) / tval!(8, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * theta.cos()
            * (tval!(3, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_4^{-2}(θ, φ).
    pub fn sph_y42s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(45, usize) / tval!(16, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(7, usize) * theta.cos().powi(2) - T::one())
            * (tval!(2, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_4^{-1}(θ, φ).
    pub fn sph_y41s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(45, usize) / tval!(8, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(7, usize) * theta.clone().cos().powi(3) - tval!(3, usize) * theta.cos())
            * varphi.sin()
    }

    /// Real spherical harmonic Y_4^0(θ, φ).
    pub fn sph_y40<T>(theta: T, _varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(9, usize) / tval!(64, usize)).sqrt()
            * (tval!(35, usize) * theta.clone().cos().powi(4)
                - tval!(30, usize) * theta.clone().cos().powi(2)
                + tval!(3, usize))
    }

    /// Real spherical harmonic Y_4^1(θ, φ).
    pub fn sph_y41c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(45, usize) / tval!(8, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(7, usize) * theta.clone().cos().powi(3) - tval!(3, usize) * theta.cos())
            * varphi.cos()
    }

    /// Real spherical harmonic Y_4^2(θ, φ).
    pub fn sph_y42c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(45, usize) / tval!(16, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(7, usize) * theta.cos().powi(2) - T::one())
            * (tval!(2, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_4^3(θ, φ).
    pub fn sph_y43c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(315, usize) / tval!(8, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * theta.cos()
            * (tval!(3, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_4^4(θ, φ).
    pub fn sph_y44c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(315, usize) / tval!(64, usize)).sqrt()
            * theta.sin().powi(4)
            * (tval!(4, usize) * varphi).cos()
    }

    // l=5 functions

    /// Real spherical harmonic Y_5^{-5}(θ, φ).
    pub fn sph_y55s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(693, usize) / tval!(128, usize)).sqrt()
            * theta.sin().powi(5)
            * (tval!(5, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_5^{-4}(θ, φ).
    pub fn sph_y54s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3465, usize) / tval!(64, usize)).sqrt()
            * theta.clone().sin().powi(4)
            * theta.cos()
            * (tval!(4, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_5^{-3}(θ, φ).
    pub fn sph_y53s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(385, usize) / tval!(128, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * (tval!(9, usize) * theta.cos().powi(2) - T::one())
            * (tval!(3, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_5^{-2}(θ, φ).
    pub fn sph_y52s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(1155, usize) / tval!(16, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(3, usize) * theta.clone().cos().powi(3) - theta.cos())
            * (tval!(2, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_5^{-1}(θ, φ).
    pub fn sph_y51s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(165, usize) / tval!(64, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(21, usize) * theta.clone().cos().powi(4)
                - tval!(14, usize) * theta.clone().cos().powi(2)
                + T::one())
            * varphi.sin()
    }

    /// Real spherical harmonic Y_5^0(θ, φ).
    pub fn sph_y50<T>(theta: T, _varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(11, usize) / tval!(64, usize)).sqrt()
            * (tval!(63, usize) * theta.clone().cos().powi(5)
                - tval!(70, usize) * theta.clone().cos().powi(3)
                + tval!(15, usize) * theta.cos())
    }

    /// Real spherical harmonic Y_5^1(θ, φ).
    pub fn sph_y51c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(165, usize) / tval!(64, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(21, usize) * theta.clone().cos().powi(4)
                - tval!(14, usize) * theta.clone().cos().powi(2)
                + T::one())
            * varphi.cos()
    }

    /// Real spherical harmonic Y_5^2(θ, φ).
    pub fn sph_y52c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(1155, usize) / tval!(16, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(3, usize) * theta.clone().cos().powi(3) - theta.cos())
            * (tval!(2, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_5^3(θ, φ).
    pub fn sph_y53c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(385, usize) / tval!(128, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * (tval!(9, usize) * theta.cos().powi(2) - T::one())
            * (tval!(3, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_5^4(θ, φ).
    pub fn sph_y54c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3465, usize) / tval!(64, usize)).sqrt()
            * theta.clone().sin().powi(4)
            * theta.cos()
            * (tval!(4, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_5^5(θ, φ).
    pub fn sph_y55c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(693, usize) / tval!(128, usize)).sqrt()
            * theta.sin().powi(5)
            * (tval!(5, usize) * varphi).cos()
    }

    // l=6 functions

    /// Real spherical harmonic Y_6^{-6}(θ, φ).
    pub fn sph_y66s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3003, usize) / tval!(512, usize)).sqrt()
            * theta.sin().powi(6)
            * (tval!(6, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_6^{-5}(θ, φ).
    pub fn sph_y65s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(9009, usize) / tval!(128, usize)).sqrt()
            * theta.clone().sin().powi(5)
            * theta.cos()
            * (tval!(5, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_6^{-4}(θ, φ).
    pub fn sph_y64s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(819, usize) / tval!(256, usize)).sqrt()
            * theta.clone().sin().powi(4)
            * (tval!(11, usize) * theta.cos().powi(2) - T::one())
            * (tval!(4, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_6^{-3}(θ, φ).
    pub fn sph_y63s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(2730, usize) / tval!(256, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * (tval!(11, usize) * theta.clone().cos().powi(3) - tval!(3, usize) * theta.cos())
            * (tval!(3, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_6^{-2}(θ, φ).
    pub fn sph_y62s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(2730, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(33, usize) * theta.clone().cos().powi(4)
                - tval!(18, usize) * theta.clone().cos().powi(2)
                + T::one())
            * (tval!(2, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_6^{-1}(θ, φ).
    pub fn sph_y61s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(273, usize) / tval!(64, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(33, usize) * theta.clone().cos().powi(5)
                - tval!(30, usize) * theta.clone().cos().powi(3)
                + tval!(5, usize) * theta.cos())
            * varphi.sin()
    }

    /// Real spherical harmonic Y_6^0(θ, φ).
    pub fn sph_y60<T>(theta: T, _varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(13, usize) / tval!(256, usize)).sqrt()
            * (tval!(231, usize) * theta.clone().cos().powi(6)
                - tval!(315, usize) * theta.clone().cos().powi(4)
                + tval!(105, usize) * theta.clone().cos().powi(2)
                - tval!(5, usize))
    }

    /// Real spherical harmonic Y_6^1(θ, φ).
    pub fn sph_y61c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(273, usize) / tval!(64, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(33, usize) * theta.clone().cos().powi(5)
                - tval!(30, usize) * theta.clone().cos().powi(3)
                + tval!(5, usize) * theta.cos())
            * varphi.cos()
    }

    /// Real spherical harmonic Y_6^2(θ, φ).
    pub fn sph_y62c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(2730, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(33, usize) * theta.clone().cos().powi(4)
                - tval!(18, usize) * theta.clone().cos().powi(2)
                + T::one())
            * (tval!(2, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_6^3(θ, φ).
    pub fn sph_y63c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(2730, usize) / tval!(256, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * (tval!(11, usize) * theta.clone().cos().powi(3) - tval!(3, usize) * theta.cos())
            * (tval!(3, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_6^4(θ, φ).
    pub fn sph_y64c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(819, usize) / tval!(256, usize)).sqrt()
            * theta.clone().sin().powi(4)
            * (tval!(11, usize) * theta.cos().powi(2) - T::one())
            * (tval!(4, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_6^5(θ, φ).
    pub fn sph_y65c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(9009, usize) / tval!(128, usize)).sqrt()
            * theta.clone().sin().powi(5)
            * theta.cos()
            * (tval!(5, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_6^6(θ, φ).
    pub fn sph_y66c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3003, usize) / tval!(512, usize)).sqrt()
            * theta.sin().powi(6)
            * (tval!(6, usize) * varphi).cos()
    }

    // l=7 functions

    /// Real spherical harmonic Y_7^{-7}(θ, φ).
    pub fn sph_y77s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(6435, usize) / tval!(1024, usize)).sqrt()
            * theta.sin().powi(7)
            * (tval!(7, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_7^{-6}(θ, φ).
    pub fn sph_y76s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(45045, usize) / tval!(512, usize)).sqrt()
            * theta.clone().sin().powi(6)
            * theta.cos()
            * (tval!(6, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_7^{-5}(θ, φ).
    pub fn sph_y75s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3465, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(5)
            * (tval!(13, usize) * theta.cos().powi(2) - T::one())
            * (tval!(5, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_7^{-4}(θ, φ).
    pub fn sph_y74s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3465, usize) / tval!(256, usize)).sqrt()
            * theta.clone().sin().powi(4)
            * (tval!(13, usize) * theta.clone().cos().powi(3) - tval!(3, usize) * theta.cos())
            * (tval!(4, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_7^{-3}(θ, φ).
    pub fn sph_y73s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(315, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * (tval!(143, usize) * theta.clone().cos().powi(4)
                - tval!(66, usize) * theta.clone().cos().powi(2)
                + tval!(3, usize))
            * (tval!(3, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_7^{-2}(θ, φ).
    pub fn sph_y72s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(315, usize) / tval!(512, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(143, usize) * theta.clone().cos().powi(5)
                - tval!(110, usize) * theta.clone().cos().powi(3)
                + tval!(15, usize) * theta.cos())
            * (tval!(2, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_7^{-1}(θ, φ).
    pub fn sph_y71s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(105, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(429, usize) * theta.clone().cos().powi(6)
                - tval!(495, usize) * theta.clone().cos().powi(4)
                + tval!(135, usize) * theta.clone().cos().powi(2)
                - tval!(5, usize))
            * varphi.sin()
    }

    /// Real spherical harmonic Y_7^0(θ, φ).
    pub fn sph_y70<T>(theta: T, _varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(15, usize) / tval!(256, usize)).sqrt()
            * (tval!(429, usize) * theta.clone().cos().powi(7)
                - tval!(693, usize) * theta.clone().cos().powi(5)
                + tval!(315, usize) * theta.clone().cos().powi(3)
                - tval!(35, usize) * theta.cos())
    }

    /// Real spherical harmonic Y_7^1(θ, φ).
    pub fn sph_y71c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(105, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin()
            * (tval!(429, usize) * theta.clone().cos().powi(6)
                - tval!(495, usize) * theta.clone().cos().powi(4)
                + tval!(135, usize) * theta.clone().cos().powi(2)
                - tval!(5, usize))
            * varphi.cos()
    }

    /// Real spherical harmonic Y_7^2(θ, φ).
    pub fn sph_y72c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(315, usize) / tval!(512, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(143, usize) * theta.clone().cos().powi(5)
                - tval!(110, usize) * theta.clone().cos().powi(3)
                + tval!(15, usize) * theta.cos())
            * (tval!(2, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_7^3(θ, φ).
    pub fn sph_y73c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(315, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * (tval!(143, usize) * theta.clone().cos().powi(4)
                - tval!(66, usize) * theta.clone().cos().powi(2)
                + tval!(3, usize))
            * (tval!(3, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_7^4(θ, φ).
    pub fn sph_y74c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3465, usize) / tval!(256, usize)).sqrt()
            * theta.clone().sin().powi(4)
            * (tval!(13, usize) * theta.clone().cos().powi(3) - tval!(3, usize) * theta.cos())
            * (tval!(4, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_7^5(θ, φ).
    pub fn sph_y75c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3465, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(5)
            * (tval!(13, usize) * theta.cos().powi(2) - T::one())
            * (tval!(5, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_7^6(θ, φ).
    pub fn sph_y76c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(45045, usize) / tval!(512, usize)).sqrt()
            * theta.clone().sin().powi(6)
            * theta.cos()
            * (tval!(6, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_7^7(θ, φ).
    pub fn sph_y77c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(6435, usize) / tval!(1024, usize)).sqrt()
            * theta.sin().powi(7)
            * (tval!(7, usize) * varphi).cos()
    }

    // l=8 functions

    /// Real spherical harmonic Y_8^{-8}(θ, φ).
    pub fn sph_y88s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(109395, usize) / tval!(16384, usize)).sqrt()
            * theta.sin().powi(8)
            * (tval!(8, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_8^{-7}(θ, φ).
    pub fn sph_y87s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(109395, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(7)
            * theta.cos()
            * (tval!(7, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_8^{-6}(θ, φ).
    pub fn sph_y86s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(7293, usize) / tval!(2048, usize)).sqrt()
            * theta.clone().sin().powi(6)
            * (tval!(15, usize) * theta.cos().powi(2) - T::one())
            * (tval!(6, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_8^{-5}(θ, φ).
    pub fn sph_y85s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(17017, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(5)
            * (tval!(15, usize) * theta.clone().cos().powi(3) - tval!(3, usize) * theta.cos())
            * (tval!(5, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_8^{-4}(θ, φ).
    pub fn sph_y84s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(11781, usize) / tval!(4096, usize)).sqrt()
            * theta.clone().sin().powi(4)
            * (tval!(65, usize) * theta.clone().cos().powi(4)
                - tval!(26, usize) * theta.clone().cos().powi(2)
                + T::one())
            * (tval!(4, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_8^{-3}(θ, φ).
    pub fn sph_y83s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(19635, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * (tval!(39, usize) * theta.clone().cos().powi(5)
                - tval!(26, usize) * theta.clone().cos().powi(3)
                + tval!(3, usize) * theta.cos())
            * (tval!(3, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_8^{-2}(θ, φ).
    pub fn sph_y82s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(5355, usize) / tval!(2048, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(143, usize) * theta.clone().cos().powi(6)
                - tval!(143, usize) * theta.clone().cos().powi(4)
                + tval!(33, usize) * theta.clone().cos().powi(2)
                - T::one())
            * (tval!(2, usize) * varphi).sin()
    }

    /// Real spherical harmonic Y_8^{-1}(θ, φ).
    pub fn sph_y81s<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3, usize) / tval!(32, usize))
            * (tval!(17, usize)).sqrt()
            * theta.clone().cos()
            * (tval!(385, usize) * theta.clone().cos().powi(2)
                - tval!(1001, usize) * theta.clone().cos().powi(4)
                + tval!(715, usize) * theta.clone().cos().powi(6)
                - tval!(35, usize))
            * theta.sin()
            * varphi.sin()
    }

    /// Real spherical harmonic Y_8^0(θ, φ).
    pub fn sph_y80<T>(theta: T, _varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(17, usize) / tval!(16384, usize)).sqrt()
            * (tval!(35, usize) - tval!(1260, usize) * theta.clone().cos().powi(2)
                + tval!(6930, usize) * theta.clone().cos().powi(4)
                - tval!(12012, usize) * theta.clone().cos().powi(6)
                + tval!(6435, usize) * theta.cos().powi(8))
    }

    /// Real spherical harmonic Y_8^1(θ, φ).
    pub fn sph_y81c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(3, usize) / tval!(32, usize))
            * (tval!(17, usize)).sqrt()
            * theta.clone().cos()
            * (tval!(385, usize) * theta.clone().cos().powi(2)
                - tval!(1001, usize) * theta.clone().cos().powi(4)
                + tval!(715, usize) * theta.clone().cos().powi(6)
                - tval!(35, usize))
            * theta.sin()
            * varphi.cos()
    }

    /// Real spherical harmonic Y_8^2(θ, φ).
    pub fn sph_y82c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(5355, usize) / tval!(2048, usize)).sqrt()
            * theta.clone().sin().powi(2)
            * (tval!(143, usize) * theta.clone().cos().powi(6)
                - tval!(143, usize) * theta.clone().cos().powi(4)
                + tval!(33, usize) * theta.clone().cos().powi(2)
                - T::one())
            * (tval!(2, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_8^3(θ, φ).
    pub fn sph_y83c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(19635, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(3)
            * (tval!(39, usize) * theta.clone().cos().powi(5)
                - tval!(26, usize) * theta.clone().cos().powi(3)
                + tval!(3, usize) * theta.cos())
            * (tval!(3, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_8^4(θ, φ).
    pub fn sph_y84c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(11781, usize) / tval!(4096, usize)).sqrt()
            * theta.clone().sin().powi(4)
            * (tval!(65, usize) * theta.clone().cos().powi(4)
                - tval!(26, usize) * theta.clone().cos().powi(2)
                + T::one())
            * (tval!(4, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_8^5(θ, φ).
    pub fn sph_y85c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(17017, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(5)
            * (tval!(15, usize) * theta.clone().cos().powi(3) - tval!(3, usize) * theta.cos())
            * (tval!(5, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_8^6(θ, φ).
    pub fn sph_y86c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(7293, usize) / tval!(2048, usize)).sqrt()
            * theta.clone().sin().powi(6)
            * (tval!(15, usize) * theta.cos().powi(2) - T::one())
            * (tval!(6, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_8^7(θ, φ).
    pub fn sph_y87c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(109395, usize) / tval!(1024, usize)).sqrt()
            * theta.clone().sin().powi(7)
            * theta.cos()
            * (tval!(7, usize) * varphi).cos()
    }

    /// Real spherical harmonic Y_8^8(θ, φ).
    pub fn sph_y88c<T>(theta: T, varphi: T) -> T
    where
        T: RealField,
    {
        (tval!(109395, usize) / tval!(16384, usize)).sqrt()
            * theta.sin().powi(8)
            * (tval!(8, usize) * varphi).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;

    /// Compute orthogonality integral on the 2-sphere surface.
    fn ortho_integral(f: impl Fn(f64, f64) -> f64, g: impl Fn(f64, f64) -> f64) -> f64 {
        let n_theta = 200;
        let n_varphi = 200;
        let d_theta = std::f64::consts::PI / n_theta as f64;
        let d_varphi = 2.0 * std::f64::consts::PI / n_varphi as f64;

        let mut integral = 0.0;

        for i in 0..n_theta {
            let theta = (i as f64 + 0.5) * d_theta;
            for j in 0..n_varphi {
                let varphi = (j as f64 + 0.5) * d_varphi;
                integral += f(theta, varphi) * g(theta, varphi) * theta.sin() * d_theta * d_varphi;
            }
        }
        integral
    }

    #[test]
    fn test_sph_get_lm_basic() {
        // Test the expected ordering: (0,0), (1,-1), (1,0), (1,1), (2,-2), ...
        assert_eq!(sph_get_lm(0), (0, 0));
        assert_eq!(sph_get_lm(1), (1, -1));
        assert_eq!(sph_get_lm(2), (1, 0));
        assert_eq!(sph_get_lm(3), (1, 1));
        assert_eq!(sph_get_lm(4), (2, -2));
        assert_eq!(sph_get_lm(5), (2, -1));
        assert_eq!(sph_get_lm(6), (2, 0));
        assert_eq!(sph_get_lm(7), (2, 1));
        assert_eq!(sph_get_lm(8), (2, 2));
    }

    #[test]
    fn test_sph_get_index_basic() {
        // Test the inverse mapping
        assert_eq!(sph_get_index(0, 0), Some(0));
        assert_eq!(sph_get_index(1, -1), Some(1));
        assert_eq!(sph_get_index(1, 0), Some(2));
        assert_eq!(sph_get_index(1, 1), Some(3));
        assert_eq!(sph_get_index(2, -2), Some(4));
        assert_eq!(sph_get_index(2, -1), Some(5));
        assert_eq!(sph_get_index(2, 0), Some(6));
        assert_eq!(sph_get_index(2, 1), Some(7));
        assert_eq!(sph_get_index(2, 2), Some(8));
    }

    #[test]
    fn test_sph_get_lm_and_index_inverse() {
        // Test that the functions are true inverses for all l up to 10
        for l in 0..11 {
            for m in (-(l as isize))..=(l as isize) {
                let index = sph_get_index(l, m).expect("valid (l, m) should produce index");
                let (l_recovered, m_recovered) = sph_get_lm(index);
                assert_eq!(
                    (l, m),
                    (l_recovered, m_recovered),
                    "Round-trip failed for (l={}, m={}): got ({}, {})",
                    l,
                    m,
                    l_recovered,
                    m_recovered
                );
            }
        }
    }

    #[test]
    fn test_sph_get_index_invalid_m() {
        // Test that invalid m values return None
        assert_eq!(sph_get_index(2, -3), None, "m < -l should return None");
        assert_eq!(sph_get_index(2, 3), None, "m > l should return None");
    }

    #[test]
    fn test_spherical_harmonics_orthogonality() {
        for l1 in 0..9 {
            for m1 in -l1..(l1 + 1) {
                for l2 in 0..9 {
                    for m2 in -l2..(l2 + 1) {
                        let integral =
                            ortho_integral(sph_ylm(l1 as usize, m1), sph_ylm(l2 as usize, m2));

                        if l1 == l2 && m1 == m2 {
                            assert!(
                                ulps_eq!(
                                    integral,
                                    4.0 * std::f64::consts::PI,
                                    max_ulps = 5,
                                    epsilon = 1e-2
                                ),
                                "Orthogonality failed: Y_{}^{} (integral = {})",
                                l1,
                                m1,
                                integral
                            );
                        } else {
                            assert!(
                                ulps_eq!(integral, 0.0, max_ulps = 5, epsilon = 1e-2),
                                "Orthogonality failed: Y_{}^{} not orthogonal to Y_{}^{} (integral = {})",
                                l1,
                                m1,
                                l2,
                                m2,
                                integral
                            );
                        }
                    }
                }
            }
        }
    }
}
