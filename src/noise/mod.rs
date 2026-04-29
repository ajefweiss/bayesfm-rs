//! Noise models & types.
//!
//! Each noise model is associated with a specific type of observation data type `OD`.

mod vector;

pub use vector::*;

use crate::obs::Obs;
use nalgebra::DVectorViewMut;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

/// A trait that is shared by all observation noise models.
pub trait Noise<OD>: Sync
where
    OD: Obs,
{
    /// Generate a random noise time-series.
    fn add_noise(&self, data: &mut DVectorViewMut<OD>, rng: &mut impl RngExt);

    /// Get random number seed.
    fn get_random_seed(&self) -> u64;

    /// Increment random number seed.
    fn increment_random_seed(&mut self);

    /// Initialize a new random number generator using the base seed.
    fn initialize_rng(&self, multiplier: u64, offset: u64) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(self.get_random_seed() * multiplier + offset)
    }
}

/// A noise model that does nothing.
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct NullNoise {}

impl<OD> Noise<OD> for NullNoise
where
    OD: Obs,
{
    fn add_noise(&self, _data: &mut DVectorViewMut<OD>, _rng: &mut impl RngExt) {}

    fn get_random_seed(&self) -> u64 {
        0
    }

    fn increment_random_seed(&mut self) {}
}
