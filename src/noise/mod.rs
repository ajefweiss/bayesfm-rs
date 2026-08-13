//! Noise models & types.
//!
//! Each noise model is associated with a specific type of observation data type `OD`.

mod vector;

pub use vector::*;

use crate::obs::Obs;
use nalgebra::DVectorViewMut;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

/// A trait that is shared by all observation noise models.
pub trait Noise<OD>: Sync
where
    OD: Obs,
{
    /// Generate a random noise time-series.
    fn add_noise<R>(&self, data: &mut DVectorViewMut<OD>, rng: &mut R)
    where
        R: RngExt + SeedableRng;
}

/// A noise model that does nothing.
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct NullNoise {}

impl<OD> Noise<OD> for NullNoise
where
    OD: Obs,
{
    fn add_noise<R>(&self, _data: &mut DVectorViewMut<OD>, _rng: &mut R)
    where
        R: RngExt + SeedableRng,
    {
    }
}
