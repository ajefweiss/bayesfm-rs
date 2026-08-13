//! bayesfm - A Bayesian forward modeling framework

pub mod conf;
mod ensbl;
pub mod geometry;
mod macros;
pub mod math;
pub mod methods;
pub mod noise;
pub mod obs;

pub use ensbl::{EnsembleModel, EnsembleObservations, EnsembleState};

use crate::{
    conf::{ConfPosition, ConfSeries},
    geometry::Geometry,
    obs::{Obs, ObsCoordBasis},
};
use itertools::zip_eq;
use nalgebra::{Const, DVector, RealField, SVectorView, SVectorViewMut, U1};
use prodef::{Density, Domain};
use rand::{RngExt, SeedableRng};
use std::ops::Sub;
use thiserror::Error;

// Enable access to the Python types if the "pyo3" feature is enabled.
#[cfg(feature = "pyo3")]
pub mod pytypes;

/// A trait that is shared by all Bayesian forward models within the crate's framework.
///
/// # Type Parameters
/// - `T`: Numeric field type (e.g., f32 or f64 implementing `RealField`)
/// - `D`: Number of spatial dimensions (const generic)
/// - `P`: Number of model parameters (const generic)
///
/// # Associated Types
/// - `FMST`: Forward model state type (can evolve over time via `evolve_fmst()`)
///
/// # Key Methods
/// - `evolve_fmst()`: Time-step the forward model state
/// - `initialize()`: Initialize parameters and both states from prior
/// - `simulate()`: Run forward simulation and generate synthetic observations
/// - `prior_density()`: Evaluate prior probability density at given parameters
/// - `prior_domain()`: Return the valid parameter domain
pub trait Model<T, const D: usize, const P: usize>: Geometry<T, D, P>
where
    T: RealField,
    Self: Sync,
{
    /// Flag that allows/disallows negative time steps.
    const ALLOW_NEGATIVE_TIMESTEPS: bool = false;

    /// Associated forward model state type.
    type FMST: Clone + Default + Send;

    /// Evolve the forward model state over a time interval.
    ///
    /// This method advances the forward model state over the given time step.
    /// It is typically called between observations to propagate model state forward in time.
    ///
    /// # Parameters
    ///
    /// - **time_step**: Duration to advance (units depend on model)
    ///   - Semantics: Advance internal model time by this amount (relative, not absolute)
    ///   - Behavior: Modifies `fm_state` in-place; time increment is additive
    ///
    /// - **params**: Parameter vector (immutable view)
    ///   - Used to control state evolution (e.g., model coefficients)
    ///   - Not modified by this method
    ///
    /// - **fm_state**: Forward model state (mutable, in-place modification)
    ///   - Before: State at time t
    ///   - After: State at time t + time_step
    ///   - Must be valid and initialized before calling
    ///
    /// - **cs_state**: Coordinate system state (mutable, may be updated)
    ///   - May affect state evolution (e.g., rotating frame)
    ///   - Can be modified by this method if coordinate system changes
    ///
    /// # Behavior
    ///
    /// This is an **additive time step**: if called multiple times with time_steps `[Δt1, Δt2, ...]`,
    /// the cumulative effect is equivalent to a single call with `Δt_total = Δt1 + Δt2 + ...`.
    ///
    /// **Important:** Multiple calls with the same time_step should produce the same result
    /// regardless of when they're called (deterministic evolution).
    ///
    /// # Determinism
    ///
    /// Must be deterministic: same inputs always produce same output.
    /// No RNG usage allowed (state evolution is deterministic).
    ///
    /// # Errors
    ///
    /// Returns `Err(ModelError)` if:
    /// - State evolution is invalid (e.g., unphysical parameters)
    /// - Coordinate transform failed
    /// - Numerical instability detected
    fn evolve_fmst(
        &self,
        time_step: T,
        params: &SVectorView<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<(), ModelError<T>>;

    /// Initialize the model parameters, and both the coordinate system and forward model states.
    fn initialize<R, G>(
        &self,
        params: &mut SVectorViewMut<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
        prior: G,
        rng: &mut R,
    ) -> Result<(), ModelError<T>>
    where
        R: RngExt + SeedableRng,
        G: Density<T, Const<P>>,
    {
        self.initialize_params::<R, G>(params, prior, rng)?;
        self.initialize_states(&params.as_view(), fm_state, cs_state)?;

        Ok(())
    }

    /// Initialize the model parameters.
    fn initialize_params<R, G>(
        &self,
        params: &mut SVectorViewMut<T, P>,
        prior: G,
        rng: &mut R,
    ) -> Result<(), ModelError<T>>
    where
        R: RngExt + SeedableRng,
        G: Density<T, Const<P>>,
    {
        let sampled_params = prior.sample(rng);

        if let Some(param_col) = sampled_params {
            params.set_column(0, &param_col);

            Ok(())
        } else {
            Err(ModelError::Sampling)
        }
    }

    /// Initialize the coordinate system and forward model states.
    fn initialize_states(
        &self,
        params: &SVectorView<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<(), ModelError<T>>;

    /// Return the model prior.
    fn prior(&self) -> impl Density<T, Const<P>> + 'static;

    /// Return the density of the model prior for a given parameter sample.
    fn prior_density(&self, params: &SVectorView<T, P>) -> Option<T>;

    /// Return the underlying function domain of the model prior.
    fn prior_domain(&self) -> Domain<T, Const<P>>;

    /// Perform a forward simulation and generate synthetic observables `OD` for the
    /// given spacecraft observers, with configuration type `OC`, for a given generating function `OF`.
    fn simulate<OC, OD, OF>(
        &self,
        obs: (&OC, &ConfSeries<OC>),
        params: &SVectorView<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
        obs_func: &OF,
    ) -> Result<DVector<OD>, ModelError<T>>
    where
        OC: Default,
        for<'a> &'a OC: Sub<&'a OC, Output = T>,
        OD: Obs,
        OF: Fn(
            &Self,
            &OC,
            &SVectorView<T, P>,
            &Self::FMST,
            &Self::CSST,
        ) -> Result<OD, ModelError<T>>,
    {
        let mut last_observation = obs.0;

        let mut obs_ensbl_vector = DVector::zeros(obs.1.len());

        zip_eq(obs.1, obs_ensbl_vector.iter_mut()).try_for_each(|(conf, obs)| {
            // Compute time step to next observation.
            let time_step = conf - last_observation;
            last_observation = conf;

            if !Self::ALLOW_NEGATIVE_TIMESTEPS && time_step < T::zero() {
                return Err(ModelError::Evolution(time_step));
            } else {
                self.evolve_fmst(time_step, params, fm_state, cs_state)?;

                *obs = obs_func(self, conf, params, fm_state, cs_state)?;
            }

            Ok::<(), ModelError<T>>(())
        })?;

        Ok(obs_ensbl_vector)
    }

    /// Perform a forward simulation and return the basis vectors for the
    /// given spacecraft observers, with configuration type `OC`.
    fn simulate_basis<OC>(
        &self,
        obs: (&OC, &ConfSeries<OC>),
        params: &SVectorView<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<DVector<ObsCoordBasis<T, D>>, ModelError<T>>
    where
        OC: Default + ConfPosition<T, D>,
        for<'a> &'a OC: Sub<&'a OC, Output = T>,
    {
        let mut last_observation = obs.0;

        let mut obs_ensbl_vector = DVector::zeros(obs.1.len());

        zip_eq(obs.1, obs_ensbl_vector.iter_mut()).try_for_each(|(conf, obs)| {
            // Compute time step to next observation.
            let time_step = conf - last_observation;
            last_observation = conf;

            if !Self::ALLOW_NEGATIVE_TIMESTEPS && time_step < T::zero() {
                return Err(ModelError::Evolution(time_step));
            } else {
                self.evolve_fmst(time_step, params, fm_state, cs_state)?;

                let position = conf.position();

                let q = match Self::transform_external_to_internal::<U1, Const<D>, _, _>(
                    &position.as_view(),
                    params,
                    cs_state,
                ) {
                    Some(value) => value,
                    None => {
                        return Err(ModelError::Coordinates(position.as_slice().to_vec()));
                    }
                };

                let basis = match Self::contravariant_basis(
                    &q.as_view::<Const<D>, U1, U1, Const<D>>(),
                    params,
                    cs_state,
                ) {
                    Some(vectors) => vectors,
                    None => return Err(ModelError::Coordinates(position.as_slice().to_vec())),
                };

                *obs = ObsCoordBasis::new(q, basis);
            }

            Ok::<(), ModelError<T>>(())
        })?;

        Ok(obs_ensbl_vector)
    }
}

/// Error types associated with the [`Model`] trait.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ModelError<T> {
    #[error("failed to convert external to internal coords")]
    Coordinates(Vec<T>),
    #[error("failed to evolve the model state (dt={0:.2}sec)")]
    Evolution(T),
    #[error("length mismatch between observed and simulated data")]
    LengthMismatch,
    #[error("failed to sample from the probability density function")]
    Sampling,
}
