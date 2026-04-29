use crate::{
    EnsembleModel, ModelError,
    math::{normalize, quantiles},
    methods::filters::{FilterError, ParticleFilter},
    noise::{Noise, NullNoise},
    obs::ObsVec,
    tval,
};
use log::debug;
use nalgebra::{Const, RealField, SVector, SVectorView, Scalar, U1};
use num_traits::AsPrimitive;
use prodef::{Domain, MultivariateNormalDensity, ParticleDensity};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use std::{iter::Sum, ops::Sub, time::Instant};

impl<T, OC, M, const D: usize, const P: usize, const N: usize>
    ParticleFilter<T, OC, ObsVec<T, N>, M, D, P>
where
    M: EnsembleModel<T, D, P> + Sync,
    T: AsPrimitive<f64> + Copy + RealField + SampleUniform + Sum,
    OC: Scalar + Sync,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an approximate Bayesian Computation particle filter algorithm using a multinormal kernel.
    pub fn abc_mvnk<EF, OF, NM>(
        &mut self,
        err_func: (&EF, T),
        obs_func: &OF,
        noise: &mut NM,
    ) -> Result<T, FilterError<T>>
    where
        EF: Fn(&[ObsVec<T, N>], &[ObsVec<T, N>]) -> T + Sync,
        OF: Fn(
                &M,
                &OC,
                &SVectorView<T, P>,
                &M::FMST,
                &M::CSST,
            ) -> Result<ObsVec<T, N>, ModelError<T>>
            + Sync,
        NM: Noise<ObsVec<T, N>>,
    {
        let start = Instant::now();

        let flt_func = |arg1: &[ObsVec<T, N>], arg2: &[ObsVec<T, N>]| {
            let value = err_func.0(arg1, arg2);

            (value < err_func.1, value)
        };

        // Preserve the previous particles.
        let old_params = self.ensbl.params().clone_owned();
        let old_weights = match self.ensbl.opt_weights() {
            Some(weights) => weights.clone(),
            None => vec![T::one() / tval!(self.ensbl.len(), usize); self.ensbl.len()],
        };

        let mut mvnk = MultivariateNormalDensity::from_vectors::<U1, Const<P>>(
            &old_params.as_view(),
            Domain::new_udomain(Const::<P>),
            self.ensbl.opt_weights().map(|value| &**value),
        )
        .unwrap()
            * self.settings.exploration_factor;

        mvnk.mean = SVector::zeros();

        let ptpdf = ParticleDensity::from_vectors::<U1, Const<P>>(
            &old_params.as_view(),
            self.model.prior_domain(),
            Some(&old_weights),
            Some(mvnk),
        )
        .unwrap();

        let (filter_values, iterations) =
            match self.filter(&flt_func, obs_func, &ptpdf, &mut Some(noise), 6361) {
                Ok(result) => result,
                Err(err) => {
                    // Restore previous particles.
                    self.ensbl.set_params(&old_params.as_view());

                    return Err(err);
                }
            };

        let transitions = ptpdf.transition_weights(&self.ensbl.params());
        let new_weights = normalize(
            &self
                .ensbl
                .params()
                .par_column_iter()
                .zip(transitions.par_iter())
                .map(|(params, transition)| {
                    self.model.prior_density(&params).unwrap() * *transition
                })
                .collect::<Vec<T>>(),
        );

        // Compute the effective sample size.
        let ess = T::one() / new_weights.iter().map(|value| value.powi(2)).sum::<T>();

        self.ensbl.set_opt_weights(Some(new_weights));

        if ess < tval!(self.ensbl.len(), usize) * self.settings.effective_particle_threshold_factor
        {
            // Replace underlying particle density with previous particle density.
            self.ensbl.set_params(&old_params.as_view());
            self.ensbl.set_opt_weights(Some(old_weights.clone()));

            return Err(FilterError::EffectiveParticles(ess));
        }

        // Compute quantiles for logging purposes.
        let quantiles = quantiles(
            &filter_values,
            &[tval!(0.34, f64), tval!(0.50, f64), tval!(0.68, f64)],
        );

        let (q_low, q_mid, q_hgh) = (quantiles[0], quantiles[1], quantiles[2]);

        debug!(
            "abc_iter\n\teps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}",
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
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            ess,
            self.ensbl.len(),
        );

        self.model.initialize_states_ensbl(&mut self.ensbl)?;

        self.model.simulate_ensbl(
            &mut self.ensbl,
            &mut self.obs_ensbl,
            obs_func,
            &mut None::<&mut NullNoise>,
        )?;

        self.errors = self.obs_ensbl.errors_func(err_func.0);

        self.iterations += 1;
        self.total_runs +=
            iterations * self.ensbl.len() * self.settings.simulation_ensemble_size_factor;

        Ok(ess)
    }

    /// A loop of approximate Bayesian Computation particle filtering steps with various aborting criteria.
    pub fn abc_mvnk_loop<NM, EF, OF>(
        &mut self,
        err_func: &EF,
        obs_func: &OF,
        error_quantile: T,
        noise: &mut NM,
    ) -> Result<(Vec<T>, Vec<T>), FilterError<T>>
    where
        T: AsPrimitive<usize>,
        NM: Noise<ObsVec<T, N>>,
        EF: Fn(&[ObsVec<T, N>], &[ObsVec<T, N>]) -> T + Sync,
        OF: Fn(
                &M,
                &OC,
                &SVectorView<T, P>,
                &M::FMST,
                &M::CSST,
            ) -> Result<ObsVec<T, N>, ModelError<T>>
            + Sync,
    {
        let mut ess = Vec::new();
        let mut eps = Vec::new();

        debug!(
            "abc_loop starting, maximum {} iterations",
            self.settings.max_iterations
        );

        for _ in 0..self.settings.max_iterations {
            let threshold = self.error_quantile(error_quantile).unwrap();

            let result = self.abc_mvnk((err_func, threshold), obs_func, noise);

            match result {
                Ok(new_ess) => {
                    ess.push(new_ess);
                    eps.push(threshold)
                }
                Err(err) => return Err(err),
            }
        }

        Ok((ess, eps))
    }
}
