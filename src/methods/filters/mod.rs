//! Particle filtering algorithms & methods.

mod abc;
mod dev;
mod sir;

use crate::{
    ConfSeries, EnsembleModel, EnsembleObservations, EnsembleState, ModelError,
    math::quantiles,
    noise::{Noise, NullNoise},
    obs::Obs,
    tval,
};
use derive_builder::Builder;
use log::{debug, info};
use nalgebra::{Const, DVector, Dyn, OMatrix, RealField, SVectorView, Scalar};
use num_traits::AsPrimitive;
use prodef::{Density, SamplingMode};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    fmt::Debug,
    io::Write,
    iter::Sum,
    ops::{AddAssign, Sub},
    time::Instant,
};
use thiserror::Error;

/// Errors associated with particle filters methods.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum FilterError<T> {
    #[error("effective particle number too small")]
    EffectiveParticles(T),
    #[error("generic model error")]
    Model(#[from] ModelError<T>),
    #[error("simulations exceeded time limit {elapsed:.1} / {limit:.1} sec")]
    TimeLimit { elapsed: f64, limit: f64 },
    #[error(
        "simulations predicted to exceed time limit {elapsed:.1} / {predicted:.1} / {limit:.1} sec"
    )]
    TimeLimitPredicted {
        elapsed: f64,
        predicted: f64,
        limit: f64,
    },
}

/// Bayesian forward model particle filter for sequential inference.
///
/// This struct manages the particle filtering algorithm, maintaining an ensemble of
/// candidate parameters and iteratively refining them based on observations.
///
/// # Type Parameters
/// - `T`: Numeric type (f32 or f64) implementing `RealField`
/// - `OC`: Observer Configuration (e.g., `DateTime<Utc>`; typically scalar time)
/// - `OD`: Observation Data (e.g., `ObsVec<T, N>` or `f64`; must implement `Obs`)
/// - `M`: Forward Model type (must implement `EnsembleModel<T, D, P>`)
/// - `D`: Number of spatial dimensions (const generic)
/// - `P`: Number of model parameters (const generic)
///
/// # State Management
/// - **Ensemble:** N parameter vectors and associated model states (cached)
/// - **Observations:** Synthetic observations from latest simulation
/// - **Errors:** Particle weights from most recent iteration
/// - **Settings:** Configurable via [`ParticleFilterSettings`]
///
/// # Configuration
/// Behavior is controlled via [`ParticleFilterSettings`]:
/// - `exploration_factor`: Scales proposal kernel (2.0 recommended)
/// - `simulation_ensemble_size_factor`: Multiplier for candidate sampling
/// - `max_iterations`: Maximum ABC iterations
/// - `simulation_time_limit`: Wall-clock time limit per iteration
///
/// # Example Workflow
/// ```ignore
/// use bayesfm::methods::filters::ParticleFilter;
///
/// // Initialize filter with ensemble size 256
/// // let mut filter = ParticleFilter::new(model, ensemble, obs_config, seed, None);
///
/// // Initialize from prior (rejection-based ABC sampling)
/// // filter.initialize(&|observed, synthetic| {
/// //     (compute_error(observed, synthetic) < threshold, error)
/// // }, &observation_function, prior)?;
///
/// // Run filtering iterations
/// // for iteration in 1..=10 {
/// //     filter.dev(&error_func, &obs_func, proposal, None)?;
/// // }
///
/// // Save results
/// // filter.save("results.json5")?;
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "
    T: Serialize, 
    OC: Serialize,
    OD: Serialize,
    M: Serialize,
    M::FMST: Serialize,
    M::CSST: Serialize"))]
#[serde(bound(deserialize = "
    T: Deserialize<'de>, 
    OC: Deserialize<'de>, 
    OD: Deserialize<'de>,
    M: Deserialize<'de>,
    M::FMST: Deserialize<'de>,
    M::CSST: Deserialize<'de>"))]
pub struct ParticleFilter<T, OC, OD, M, const D: usize, const P: usize>
where
    T: RealField,
    OC: Scalar,
    OD: Scalar,
    M: EnsembleModel<T, D, P>,
    M::CSST: std::fmt::Debug + Clone,
    M::FMST: std::fmt::Debug + Clone,
{
    ensbl: EnsembleState<T, M::CSST, M::FMST, D, P>,

    errors: Vec<T>,

    iterations: usize,

    model: M,

    obs_ensbl: EnsembleObservations<OC, OD>,

    random_seed: u64,

    /// Particle filter settings.
    pub settings: ParticleFilterSettings<T>,

    total_runs: usize,
}

impl<T, OC, OD, M, const D: usize, const P: usize> ParticleFilter<T, OC, OD, M, D, P>
where
    T: Copy + RealField + Sum,
    OC: Scalar + Sync,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
    OD: AddAssign + Obs,
    M: EnsembleModel<T, D, P> + Sync,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// Return the stored errors.
    pub fn errors(&self) -> &[T] {
        &self.errors
    }

    /// Re-evaluate errors using a given error function `EF`.
    pub fn errors_func<EF>(&self, func: &EF) -> Vec<T>
    where
        EF: Fn(&[OD], &[OD]) -> T + Sync,
    {
        self.obs_ensbl.errors_func(func)
    }

    /// Compute a a quantile of the field `errors`.
    pub fn error_quantile(&self, quantile: T) -> Option<T>
    where
        T: AsPrimitive<usize>,
    {
        assert!(
            (T::zero()..T::one()).contains(&quantile),
            "quantile must be within [0, 1]"
        );

        if self.errors.is_empty() {
            None
        } else {
            let mut errors_sorted = self.errors.clone();

            errors_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            Some(errors_sorted[(quantile * tval!(self.errors.len(), usize)).as_()])
        }
    }

    /// Perform a particle filter iteration using a custom filter function.
    ///
    /// This is a generic particle filtering step that samples candidate parameters
    /// from a proposal distribution, evaluates each through the forward model, and
    /// selects particles that pass the filter function. The filter function determines
    /// which candidates are accepted as new ensemble members.
    ///
    /// # Algorithm Overview
    ///
    /// 1. Sample `factor × N` candidate parameter vectors from proposal `pdf`
    /// 2. For each candidate:
    ///    - Run forward model via `obs_func`
    ///    - Apply observation noise (if provided)
    ///    - Evaluate via `flt_func` to determine acceptance
    /// 3. Collect particles for which `flt_func` returned `true`
    /// 4. Return metric values and iteration count
    ///
    /// # Parameters
    ///
    /// ## `flt_func`: Filter Function
    /// Signature: `(&[OD], &[OD]) -> (bool, T) + Sync`
    ///
    /// Determines which candidates are accepted as ensemble members.
    /// - **Input 1**: Reference observations (immutable slice)
    /// - **Input 2**: Synthetic observations from forward model
    /// - **Output**: `(accept: bool, metric: T)`
    ///   - `accept == true`: Candidate passes filter; include in new ensemble
    ///   - `accept == false`: Candidate fails filter; reject it
    ///   - `metric`: Distance/error value for accepted particles (used for weighting/analysis)
    ///
    /// Called in parallel via Rayon; must be thread-safe (`Sync`).
    /// The metric value is stored for all accepted particles.
    ///
    /// ## `obs_func`: Observation Function
    /// Signature: `(&M, &OC, &SVectorView<T, P>, &M::FMST, &M::CSST) -> Result<OD, ModelError<T>> + Sync`
    ///
    /// Evaluates the forward model and returns synthetic observations.
    /// - **Input 1**: Model reference
    /// - **Input 2**: Observation configuration (time-series, timestamp, etc.)
    /// - **Input 3**: Parameter vector (candidate parameters)
    /// - **Input 4**: Forward model state
    /// - **Input 5**: Coordinate system state
    /// - **Output**: Synthetic observation vector
    ///
    /// Handles all forward model evaluation and error propagation.
    /// Called once per candidate per iteration; must be thread-safe.
    ///
    /// ## `pdf`: Prior or Proposal Distribution
    /// Implements: `Density<T, Const<P>> + Sync`
    ///
    /// Probability density for sampling new candidates.
    /// - First iteration: typically the prior distribution
    /// - Subsequent iterations: proposal distribution centered on previous particles
    /// - Cloned for each Rayon parallelism chunk (ensures deterministic RNG seeding)
    ///
    /// ## `opt_noise`: Optional Observation Noise
    /// Type: `Option<&mut dyn Noise<OD>>`
    ///
    /// Adds synthetic measurement noise after forward model evaluation.
    /// - `None`: No noise (use clean synthetic observations)
    /// - `Some(noise)`: Apply specified noise model
    ///
    /// Models measurement uncertainty in observations.
    ///
    /// ## `seed_multiplier`: RNG Seed Offset
    /// Used to ensure unique random sequences across iterations.
    /// - Typical usage: `seed = base_seed + iteration * seed_multiplier`
    ///
    /// # Return Value
    ///
    /// `Ok((metric_values, sub_iterations))`
    /// - **metric_values**: Metric values for each accepted particle (length = N)
    /// - **sub_iterations**: Number of rejection-sampling iterations to fill ensemble
    ///
    /// `Err(FilterError)` on timeout, excessive iterations, or model failure
    ///
    /// # Errors
    ///
    /// Returns `Err(FilterError)` if:
    /// - **TimeLimit**: Exceeded wall-clock time limit
    /// - **TimeLimitPredicted**: Predicted to exceed time limit
    /// - **EffectiveParticles**: Too few particles accepted (degeneracy)
    /// - **Model**: Forward model error (coordinate transform failed, etc.)
    pub fn filter<FF, OF, G, NM>(
        &mut self,
        flt_func: &FF,
        obs_func: &OF,
        pdf: G,
        opt_noise: &mut Option<&mut NM>,
        seed_multiplier: u64,
    ) -> Result<(Vec<T>, usize), FilterError<T>>
    where
        FF: Fn(&[OD], &[OD]) -> (bool, T) + Sync,
        OF: Fn(&M, &OC, &SVectorView<T, P>, &M::FMST, &M::CSST) -> Result<OD, ModelError<T>> + Sync,
        G: Density<T, Const<P>> + Sync,
        NM: Noise<OD>,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration: usize = 0;

        let mut target_filter_values = Vec::<T>::with_capacity(self.ensbl.len());

        // Temporary ensemble and output array.
        let (mut temp_ensbl, mut temp_obs_ensbl) = (
            EnsembleState::new(
                OMatrix::<T, Const<P>, Dyn>::zeros(
                    self.ensbl.len() * self.settings.simulation_ensemble_size_factor,
                ),
                None,
                None,
            ),
            EnsembleObservations::new(
                self.obs_ensbl.initial().clone(),
                self.obs_ensbl.configuration().clone(),
                self.ensbl.len() * self.settings.simulation_ensemble_size_factor,
                None,
            )
            .unwrap(),
        );

        // Iterate until we have enough new particles.
        while counter != self.ensbl.len() {
            self.model.initialize_ensbl(
                &mut temp_ensbl,
                pdf.clone(),
                &self.settings.sampling_mode,
                self.random_seed + seed_multiplier * iteration as u64,
            )?;

            self.model
                .simulate_ensbl(&mut temp_ensbl, &mut temp_obs_ensbl, obs_func, opt_noise)?;

            let mut filter_flags = vec![true; temp_ensbl.len()];

            let filter_values = temp_obs_ensbl
                .par_ensbl_iter()
                .zip(filter_flags.par_iter_mut())
                .chunks(M::RAYON_CHUNK_SIZE)
                .map(|mut chunk| {
                    chunk
                        .iter_mut()
                        .map(|((_, out), flag)| {
                            let (result, value) = flt_func(
                                self.obs_ensbl
                                    .ref_output()
                                    .expect("missing reference data")
                                    .as_slice(),
                                out.as_slice(),
                            );

                            **flag = result;

                            value
                        })
                        .collect::<Vec<T>>()
                })
                .flatten()
                .collect::<Vec<T>>();

            // Transform the filter flags to a list of valid indices.
            let mut indices = filter_flags
                .into_iter()
                .enumerate()
                .filter_map(|(idx, flag)| if flag { Some(idx) } else { None })
                .collect::<Vec<usize>>();

            // Remove excessive ensemble members.
            if counter + indices.len() > self.ensbl.len() {
                debug!(
                    "removing excessive ensemble members(n={})",
                    counter + indices.len() - self.ensbl.len()
                );
                indices.drain((self.ensbl.len() - counter)..indices.len());
            }

            // Copy over the results.
            indices.iter().enumerate().for_each(|(edx, idx)| {
                self.ensbl
                    .set_param(counter + edx, &temp_ensbl.params().column(*idx));

                self.obs_ensbl
                    .set_output(counter + edx, &temp_obs_ensbl.output(*idx));

                target_filter_values.push(filter_values[*idx]);
            });

            counter += indices.len();

            self.random_seed += 1;
            iteration += 1;

            // Abort if simulation time is above the given limit (or is estimated to be above).
            if start.elapsed().as_millis() as f64 / 1e3 > self.settings.simulation_time_limit {
                info!(
                    "filter aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\tcollected samples = {:.1} / {}",
                    (iteration
                        * self.ensbl.len()
                        * self.settings.simulation_ensemble_size_factor
                        * self.obs_ensbl.len()) as f64
                        / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    self.ensbl.len(),
                );

                return Err(FilterError::TimeLimit {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
                    limit: self.settings.simulation_time_limit,
                });
            } else if (counter == 0)
                || (self.settings.simulation_time_prediction
                    && ((self.ensbl.len() / counter) as f64 * start.elapsed().as_millis() as f64
                        / 1e3
                        > self.settings.simulation_time_limit))
            {
                let estimated_time = match counter.cmp(&0) {
                    Ordering::Equal => f64::INFINITY,
                    _ => {
                        (self.ensbl.len() / counter) as f64 * start.elapsed().as_millis() as f64
                            / 1e3
                    }
                };

                info!(
                    "filter pre-emptively aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\ttotal predicted duration: {:.2}\n\tcollected samples = {:.1} / {}",
                    (iteration
                        * self.ensbl.len()
                        * self.settings.simulation_ensemble_size_factor
                        * self.obs_ensbl.len()) as f64
                        / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    estimated_time,
                    counter,
                    self.ensbl.len(),
                );

                return Err(FilterError::TimeLimitPredicted {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
                    predicted: estimated_time,
                    limit: self.settings.simulation_time_limit,
                });
            }
        }

        Ok((target_filter_values, iteration))
    }

    /// Initialize the ensemble from a prior distribution `pdf` with a filtering function `FF`.
    pub fn initialize<G, FF, OF>(
        &mut self,
        flt_func: &FF,
        obs_func: &OF,
        pdf: G,
    ) -> Result<(), FilterError<T>>
    where
        T: AsPrimitive<f64>,
        G: Density<T, Const<P>> + Sync,
        FF: Fn(&[OD], &[OD]) -> (bool, T) + Sync,
        OF: Fn(&M, &OC, &SVectorView<T, P>, &M::FMST, &M::CSST) -> Result<OD, ModelError<T>> + Sync,
    {
        let start = Instant::now();

        let (filter_values, iterations) =
            self.filter(flt_func, obs_func, pdf, &mut None::<&mut NullNoise>, 7573)?;

        // Compute quantiles for logging purposes.
        let quantiles = quantiles(
            &filter_values,
            &[tval!(0.34, f64), tval!(0.50, f64), tval!(0.68, f64)],
        );

        let (q_low, q_mid, q_hgh) = (quantiles[0], quantiles[1], quantiles[2]);

        debug!(
            "initialize(n={})\n\teps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
            self.ensbl.len(),
            q_low,
            q_mid,
            q_hgh,
            tval!(
                (iterations
                    * self.ensbl.len()
                    * self.settings.simulation_ensemble_size_factor
                    * self.obs_ensbl.len()) as f64
                    / 1e6,
                f64
            ),
            tval!(start.elapsed().as_millis() as f64 / 1e3, f64),
        );

        self.errors = filter_values;

        // Set diagnostic fields.
        self.iterations = 1;
        self.total_runs =
            iterations * self.ensbl.len() * self.settings.simulation_ensemble_size_factor;

        Ok(())
    }

    /// Check if the ensemble is empty.
    pub fn is_empty(&self) -> bool {
        self.ensbl.is_empty()
    }

    /// Return the ensemble size.
    pub fn len(&self) -> usize {
        self.ensbl.len()
    }

    /// Create a new [`ParticleFilter`].
    pub fn new(
        model: M,
        ensbl: EnsembleState<T, M::CSST, M::FMST, D, P>,
        obs_ensbl: EnsembleObservations<OC, OD>,
        random_seed: u64,
        opt_settings: Option<ParticleFilterSettings<T>>,
    ) -> Self
    where
        T: Default,
    {
        Self {
            ensbl,
            errors: Vec::new(),
            iterations: 0,
            model,
            obs_ensbl,
            random_seed,
            settings: opt_settings
                .unwrap_or(ParticleFilterSettingsBuilder::default().build().unwrap()),
            total_runs: 0,
        }
    }

    /// Return the underlying model prior.
    pub fn prior(&self) -> impl Density<T, Const<P>> + 'static {
        self.model.prior()
    }

    /// Serialize result to a JSON file.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json5::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }

    /// Simulate the model ensemble and return the results in an observation ensemble.
    pub fn simulate<OF, NM>(
        &mut self,
        opt_obs: Option<&ConfSeries<OC>>,
        opt_ref_data: Option<&DVector<OD>>,
        obs_func: &OF,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<EnsembleObservations<OC, OD>, FilterError<T>>
    where
        OF: Fn(&M, &OC, &SVectorView<T, P>, &M::FMST, &M::CSST) -> Result<OD, ModelError<T>> + Sync,
        NM: Noise<OD>,
    {
        let obs_ensbl = match opt_obs {
            Some(obs) => {
                let mut obs_ensbl = EnsembleObservations::new(
                    self.obs_ensbl.initial().clone(),
                    obs.clone(),
                    self.ensbl.len(),
                    opt_ref_data.cloned(),
                )
                .unwrap();

                self.model.initialize_states_ensbl(&mut self.ensbl)?;

                self.model
                    .simulate_ensbl(&mut self.ensbl, &mut obs_ensbl, obs_func, opt_noise)?;

                obs_ensbl
            }
            None => {
                self.model.initialize_states_ensbl(&mut self.ensbl)?;

                self.model.simulate_ensbl(
                    &mut self.ensbl,
                    &mut self.obs_ensbl,
                    obs_func,
                    opt_noise,
                )?;

                self.obs_ensbl.clone()
            }
        };

        Ok(obs_ensbl)
    }

    /// Return the ensemble member weights (optional).
    pub fn weights(&self) -> Option<&Vec<T>> {
        self.ensbl.opt_weights()
    }
}

/// Configuration settings for particle filter behavior and convergence.
///
/// This struct controls key parameters of the particle filtering algorithm
/// via the builder pattern. Use `ParticleFilterSettingsBuilder::default()`
/// to create with default values, then customize specific fields.
///
/// # Creation & Usage
///
/// ```
/// use bayesfm::methods::filters::ParticleFilterSettingsBuilder;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let settings = ParticleFilterSettingsBuilder::default()
///     .exploration_factor(2.0)
///     .simulation_ensemble_size_factor(8)
///     .max_iterations(10)
///     .build()?;
/// # Ok(())
/// # }
/// ```
///
/// # Tuning Guide
///
/// ## Exploration vs. Convergence Trade-off
///
/// **exploration_factor** controls the proposal kernel width:
/// - **Low (1.0-1.5)**: Narrow proposal → fast convergence, risk of local minima
/// - **Optimal (2.0)**: Balanced exploration (Filippi et al. 2013 recommendation)
/// - **High (2.5-3.0)**: Wide proposal → slower convergence, better mode coverage
///
/// Adjust this first if filter converges prematurely or diverges.
///
/// ## Computational Budget
///
/// **simulation_ensemble_size_factor** controls how many candidates to sample:
/// - Factor × ensemble_size = total forward model runs per iteration
/// - **Low (4)**: Fast but risky (fewer accepted particles)
/// - **Typical (8)**: Balanced (default)
/// - **High (16+)**: Robust but slow
///
/// **simulation_time_limit** sets wall-clock limit per iteration:
/// - Prevents runaway on expensive models
/// - Set to very large value to effectively disable (no timeout)
///
/// ## Particle Quality
///
/// **effective_particle_threshold_factor** controls when to resample:
/// - When `N_eff / N < factor`, particles are resampled
/// - `N_eff = 1 / Σ(weight_i²)` measures degeneracy
/// - **Low (0.01)**: Aggressive resampling → maintains diversity
/// - **Typical (0.05)**: Balanced (default)
/// - **High (0.20)**: Tolerate degeneracy → less resampling overhead
///
/// # Typical Scenarios
///
/// ```
/// use bayesfm::methods::filters::ParticleFilterSettingsBuilder;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let settings = ParticleFilterSettingsBuilder::default()
///     .exploration_factor(2.0)
///     .simulation_ensemble_size_factor(8)
///     .effective_particle_threshold_factor(0.02)
///     .max_iterations(10)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Builder, Clone, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<T>
where
    T: RealField,
{
    /// Kernel scaling factor for proposal distribution.
    ///
    /// Controls the spread of the proposal distribution around the current
    /// ensemble of particles.
    ///
    /// **Recommendation:** Use 2.0 for optimal performance (Filippi et al. 2013).
    /// - `< 1.5`: Risk of premature convergence (gets stuck in local modes)
    /// - `= 2.0`: Optimal balance between exploration and convergence
    /// - `> 2.5`: Excessive exploration, slow convergence
    ///
    /// **Effect on iteration:**
    /// - Higher values: Larger jumps in parameter space, slower convergence
    /// - Lower values: Smaller jumps, faster convergence but mode-seeking
    ///
    /// **Default:** 2.0
    #[builder(default = tval!(2, usize))]
    pub exploration_factor: T,

    /// Rejection sampling configuration with fallback strategy.
    ///
    /// Controls behavior when sampling from the proposal distribution.
    /// If a sample lands outside the prior domain, retry up to max_attempts.
    ///
    /// **Default:** Up to 1024 attempts before giving up
    #[builder(default = SamplingMode::UntilValid { max_attempts: 1024 })]
    pub sampling_mode: SamplingMode,

    /// Maximum filtering iterations (failsafe limit).
    ///
    /// Each iteration samples candidates until N particles are accepted.
    /// If this limit is reached, the filter returns early with partial results.
    ///
    /// Rarely needed in well-tuned problems. Increase if getting partial results.
    ///
    /// **Default:** 10 iterations
    /// **Typical Range:** 5-20
    #[builder(default = 10)]
    pub max_iterations: usize,

    /// Resampling trigger threshold (as a fraction).
    ///
    /// When effective particle ratio `N_eff / N < this_factor`,
    /// particles are resampled to maintain ensemble diversity.
    ///
    /// Effective particle count: `N_eff = 1 / Σ(weight_i²)`
    /// - Low N_eff indicates particle weight concentration (degeneracy)
    /// - Resampling resets weights to 1/N
    ///
    /// **Tuning:**
    /// - `0.01` (1%): Aggressive, maintains diversity, higher computational cost
    /// - `0.05` (5%): Balanced (default)
    /// - `0.20` (20%): Tolerates some degeneracy, lower computational cost
    ///
    /// **Default:** 0.05
    /// **Range:** 0.0 to 1.0
    #[builder(default = tval!(0.05, f64))]
    pub effective_particle_threshold_factor: T,

    /// Multiplier for number of candidate samples per iteration.
    ///
    /// Total forward model evaluations per iteration:
    /// `n_candidates = factor × ensemble_size`
    ///
    /// Most candidates are rejected in early iterations when particles haven't
    /// converged yet. This factor controls the rejection ratio.
    ///
    /// **Interpretation:**
    /// - `4`: Sample 4N to find N good particles (aggressive)
    /// - `8`: Sample 8N to find N good particles (default, typical)
    /// - `16`: Sample 16N to find N good particles (robust)
    ///
    /// **Effect on iteration:**
    /// - Higher: Increased acceptance rate, slower iterations
    /// - Lower: Decreased acceptance rate, faster iterations but may under-sample
    ///
    /// **Default:** 8
    /// **Typical Range:** 4-16
    #[builder(default = 8)]
    pub simulation_ensemble_size_factor: usize,

    /// Wall-clock time limit per filtering iteration (seconds).
    ///
    /// If an iteration takes longer than this, abort and return partial results.
    /// Prevents runaway computations on expensive forward models.
    ///
    /// Set to large value (e.g., 1e9) to effectively disable time limit.
    ///
    /// **Default:** 5.0 seconds
    /// **Typical Range:** 1.0-60.0 seconds
    #[builder(default = 5.0)]
    pub simulation_time_limit: f64,

    /// Enable time prediction for early abort (default: true).
    ///
    /// If true, measures time for first N_sample candidates, extrapolates to
    /// full ensemble, and aborts early if predicted time exceeds time_limit.
    ///
    /// Saves computation when forward model is expensive, but may be inaccurate
    /// if evaluation time varies significantly across parameter space.
    ///
    /// **Default:** true (enabled)
    /// **Recommendation:** Keep enabled unless extrapolation is unreliable
    #[builder(default = true)]
    pub simulation_time_prediction: bool,
}
