use nalgebra::allocator::Allocator;
use nalgebra::{DVectorViewMut, DefaultAllocator, Dyn, RealField, U1};
use prodef::{Density, MultivariateNormalDensity};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::noise::Noise;
use crate::obs::ObsVec;

/// Generic N-dimensional observation vector noise models.
///
/// These variants represent different approaches to modeling measurement
/// uncertainty in observation vectors. The choice depends on the physical
/// nature of the measurement error.
///
/// # Variants
///
/// - **`AdditiveNormal(std_dev, seed)`**: Add independent Gaussian noise to each element.
///   - `std_dev`: Standard deviation of the normal distribution (absolute units)
///   - `seed`: Random number seed for reproducibility
///   - Use case: Uncorrelated sensor noise with constant magnitude across elements
///   - Formula: `obs' = obs + N(0, std_dev²)` (independent per element)
///
/// - **`AdditiveMultiNormal(cov_matrix, seed)`**: Add correlated Gaussian noise.
///   - `cov_matrix`: Covariance matrix defining correlations between elements
///   - `seed`: Random number seed for reproducibility
///   - Use case: Measurement uncertainties with spatial or temporal correlations
///   - Formula: `obs' = obs + N(0, Σ)` where Σ is the covariance matrix
///   - Useful when multiple observation channels have systematic correlations
///
/// - **`MultiplicativeNormal(std_dev, seed)`**: Scale observations by random factor (1 + ε).
///   - `std_dev`: Relative standard deviation (unitless, as a fraction of magnitude)
///   - `seed`: Random number seed for reproducibility
///   - Use case: Measurement errors that scale with observation magnitude
///   - Formula: `obs' = obs × (1 + N(0, std_dev²))`
///   - Typical for systems where error is proportional to signal strength
///
/// # Examples
///
/// ```
/// use bayesfm::noise::ObsVecNoise;
///
/// // Fixed noise level: add ±0.1 units to each observation independently
/// let noise = ObsVecNoise::<f64>::AdditiveNormal(0.1, 12345);
///
/// // Relative noise: scale observations by random factor (±5%)
/// let noise = ObsVecNoise::<f64>::MultiplicativeNormal(0.05, 67890);
///
/// // Correlated noise requires external prodef crate setup
/// // let cov_matrix = MultivariateNormalDensity::new(mean, covariance)?;
/// // let noise = ObsVecNoise::AdditiveMultiNormal(cov_matrix, 54321);
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
#[allow(missing_docs)]
pub enum ObsVecNoise<T>
where
    T: RealField,
{
    /// Independent additive Gaussian noise: `obs' = obs + N(0, std_dev²)` per element
    ///
    /// Adds independent Gaussian noise to each observation element.
    /// All elements receive noise from the same distribution (same standard deviation).
    /// Appropriate when all observations have similar, uncorrelated measurement uncertainty.
    AdditiveNormal(T, u64),

    /// Correlated additive Gaussian noise: `obs' = obs + N(0, Σ)` where Σ is covariance
    ///
    /// Adds multivariate Gaussian noise with specified covariance structure.
    /// Allows modeling dependencies and correlations between observation errors.
    /// Each noise realization is drawn from the multivariate distribution.
    AdditiveMultiNormal(MultivariateNormalDensity<T, Dyn>, u64),

    /// Multiplicative Gaussian noise: `obs' = obs × (1 + N(0, std_dev²))`
    ///
    /// Scales observations by a random multiplier (1 + ε) where ε ~ N(0, std_dev²).
    /// Models measurement errors that are proportional to the signal magnitude.
    /// Small observations get small noise; large observations get large noise.
    MultiplicativeNormal(T, u64),
}

impl<T, const N: usize> Noise<ObsVec<T, N>> for ObsVecNoise<T>
where
    T: RealField,
    DefaultAllocator: Allocator<Dyn> + Allocator<U1, Dyn> + Allocator<Dyn, Dyn>,
    StandardNormal: Distribution<T>,
{
    fn add_noise(&self, data: &mut DVectorViewMut<ObsVec<T, N>>, rng: &mut impl rand::RngExt) {
        match self {
            ObsVecNoise::AdditiveNormal(std_dev, ..) => {
                let normal = StandardNormal;

                data.iter_mut().for_each(|obs_vec| {
                    obs_vec.iter_mut().for_each(|val| {
                        *val += rng.sample(normal) * std_dev.clone();
                    })
                });
            }
            ObsVecNoise::AdditiveMultiNormal(mvnpdf, ..) => {
                for i in 0..N {
                    data.iter_mut()
                        .zip(
                            mvnpdf
                                .sample(rng, &prodef::SamplingMode::UntilValid { max_attempts: 32 })
                                .expect("failed to draw multi normal noise sample")
                                .row_iter(),
                        )
                        .for_each(|(res, val)| {
                            res.set(i, res.get(i).unwrap().clone() + val[(0, 0)].clone())
                        });
                }
            }
            ObsVecNoise::MultiplicativeNormal(std_dev, ..) => {
                let normal = StandardNormal;

                data.iter_mut().for_each(|obs_vec| {
                    obs_vec.iter_mut().for_each(|val| {
                        *val += val.clone() * rng.sample(normal) * std_dev.clone();
                    })
                });
            }
        }
    }

    fn get_random_seed(&self) -> u64 {
        match self {
            ObsVecNoise::AdditiveNormal(.., seed) => *seed,
            ObsVecNoise::AdditiveMultiNormal(.., seed) => *seed,
            ObsVecNoise::MultiplicativeNormal(.., seed) => *seed,
        }
    }

    fn increment_random_seed(&mut self) {
        match self {
            ObsVecNoise::AdditiveNormal(.., seed) => {
                *seed += 1;
            }
            ObsVecNoise::AdditiveMultiNormal(.., seed) => {
                *seed += 1;
            }
            ObsVecNoise::MultiplicativeNormal(.., seed) => {
                *seed += 1;
            }
        }
    }
}
