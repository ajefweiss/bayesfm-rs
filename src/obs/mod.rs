//! Obersvable data types.

mod basis;
mod image;
mod vector;

pub use basis::*;
pub use image::*;
pub use vector::*;

use nalgebra::Scalar;
use num_traits::Zero;

/// A trait that is shared by all model observation data types.
///
/// Types that implement this trait are commonly prefixed with `Obs` (e.g. [`ObsVec`] or [`ObsImg`]).
pub trait Obs: Clone + Default + Scalar + Send + Sync + Zero {
    /// Returns `true` if the observation is considered valid.
    fn is_valid(&self) -> bool;
}

impl Obs for f32 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

impl Obs for f64 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}
