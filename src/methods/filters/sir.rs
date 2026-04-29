use crate::{
    EnsembleModel, EnsembleObservations, EnsembleState, ModelError,
    math::normalize,
    methods::filters::{FilterError, ParticleFilter},
    noise::NullNoise,
    obs::ObsVec,
};
use itertools::Itertools;
use log::debug;
use nalgebra::{Const, DVector, Dyn, OMatrix, RealField, SVector, SVectorView, Scalar, U1};
use num_traits::AsPrimitive;
use prodef::{Density, Domain, MultivariateNormalDensity, ParticleDensity};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{cmp::Ordering, iter::Sum, ops::Sub, time::Instant};

impl<T, OC, M, const D: usize, const P: usize, const N: usize>
    ParticleFilter<T, OC, ObsVec<T, N>, M, D, P>
where
    T: Copy + RealField + SampleUniform + Sum,
    OC: Scalar + Sync,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
    M: Clone + EnsembleModel<T, D, P> + Sync,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an sequential importance resampling particle filter algorithm.
    pub fn sir_mvnk<LF, OF>(
        &mut self,
        obs_func: &OF,
        llh_func: &LF,
    ) -> Result<(T, usize), FilterError<T>>
    where
        LF: Fn(&[ObsVec<T, N>], &[ObsVec<T, N>]) -> T + Sync,
        OF: Fn(
                &M,
                &OC,
                &SVectorView<T, P>,
                &M::FMST,
                &M::CSST,
            ) -> Result<ObsVec<T, N>, ModelError<T>>
            + Sync,
    {
        let start = Instant::now();

        // Create an interim [`ParticleFilter`] with a larger ensemble size.
        let mut sub_pf = self.clone();

        sub_pf.settings.simulation_ensemble_size_factor = 1;

        sub_pf.ensbl = EnsembleState::new(
            OMatrix::<T, Const<P>, Dyn>::zeros(
                self.ensbl.len() * self.settings.simulation_ensemble_size_factor,
            ),
            None,
            None,
        );

        sub_pf.obs_ensbl = EnsembleObservations::new(
            self.obs_ensbl.initial().clone(),
            self.obs_ensbl.configuration().clone(),
            self.ensbl.len() * self.settings.simulation_ensemble_size_factor,
            Some(DVector::from_iterator(
                self.obs_ensbl.len(),
                self.obs_ensbl
                    .ref_output()
                    .expect("missing reference data")
                    .iter()
                    .cloned(),
            )),
        )
        .unwrap();

        let mut mvnk = MultivariateNormalDensity::from_vectors::<U1, Const<P>>(
            &self.ensbl.params().as_view(),
            Domain::new_udomain(Const::<P>),
            self.ensbl.opt_weights().map(|value| &**value),
        )
        .unwrap()
            * self.settings.exploration_factor;

        mvnk.mean = SVector::zeros();

        let domain = self.model.prior_domain();

        let ptpdf = ParticleDensity::from_vectors::<U1, Const<P>>(
            &self.ensbl.params().as_view(),
            domain.clone(),
            self.ensbl.opt_weights().map(|value| &**value),
            Some(mvnk),
        )
        .unwrap();

        let flt_func = |arg1: &[ObsVec<T, N>], arg2: &[ObsVec<T, N>]| {
            let value = llh_func(arg1, arg2);

            (value.is_finite(), value)
        };

        let (interim_likelihood_values, iterations) = sub_pf.filter(
            &flt_func,
            obs_func,
            &ptpdf,
            &mut None::<&mut NullNoise>,
            7901,
        )?;

        // Offset log-likelihood values to reduce precision issues.
        let llh_max = *interim_likelihood_values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
            .unwrap();

        // Convert log-likelihood to likelihood and apply prior and importance weight.
        let transitions = ptpdf.transition_weights(&sub_pf.ensbl.params());
        let interim_weights = normalize(
            &interim_likelihood_values
                .par_iter()
                .zip(sub_pf.ensbl.params().par_column_iter())
                .zip(transitions.par_iter())
                .map(|((llh, params), transition)| {
                    (*llh - llh_max).exp()
                        * self.model.prior_density(&params).unwrap()
                        * *transition
                })
                .collect::<Vec<T>>(),
        );

        // Compute the effective sample size from the interim weights
        let ess = T::one() / interim_weights.iter().map(|value| value.powi(2)).sum::<T>();

        // Update interim weights.
        sub_pf.ensbl.set_opt_weights(Some(interim_weights));

        let sub_ptpdf = ParticleDensity::from_vectors::<U1, Const<P>>(
            &sub_pf.ensbl.params().as_view(),
            domain.clone(),
            sub_pf.ensbl.opt_weights().map(|value| &**value),
            None,
        )
        .unwrap();

        self.ensbl = EnsembleState::new_resampled(
            &self.model,
            self.ensbl.len(),
            &sub_ptpdf,
            self.random_seed + 1877,
        )
        .unwrap();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.random_seed);

        let constants = sub_ptpdf
            .domain()
            .size()
            .iter()
            .map(|size| match size {
                Some(value) => value.partial_cmp(&T::zero()).unwrap() == std::cmp::Ordering::Equal,
                None => false,
            })
            .collect::<Vec<bool>>();

        let mut udx = rng.random_range(0..D);

        // Select a dimension that is not fixed.
        while constants[udx] {
            udx = rng.random_range(0..D);
        }

        let uniques = self
            .ensbl
            .params()
            .row(udx)
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .dedup()
            .copied()
            .collect::<Vec<T>>()
            .len();

        self.model.simulate_ensbl(
            &mut self.ensbl,
            &mut self.obs_ensbl,
            obs_func,
            &mut None::<&mut NullNoise>,
        )?;

        debug!(
            "sir_iter\n\tran {:2.3}M evaluations in {:.2} sec\n\tunique samples = {:.1} / {}",
            T::from_f64(
                (iterations
                    * self.ensbl.len()
                    * self.settings.simulation_ensemble_size_factor
                    * self.obs_ensbl.len()) as f64
                    / 1e6
            )
            .unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            uniques,
            self.ensbl.len(),
        );

        self.errors = self.obs_ensbl.errors_func(llh_func);

        self.iterations += 1;
        self.random_seed += 1;
        self.total_runs +=
            iterations * self.ensbl.len() * self.settings.simulation_ensemble_size_factor;

        Ok((ess, uniques))
    }

    /// A loop of sequential importance re-sampling steps with various aborting criteria.
    pub fn sir_mvnk_loop<LF, OF>(
        &mut self,
        obs_func: &OF,
        llh_func: &LF,
    ) -> Result<(Vec<T>, Vec<usize>), FilterError<T>>
    where
        LF: Fn(&[ObsVec<T, N>], &[ObsVec<T, N>]) -> T + Sync,
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
        let mut uniques = Vec::new();

        for _ in 0..self.settings.max_iterations {
            let result = self.sir_mvnk(obs_func, llh_func);

            match result {
                Ok((new_ess, new_uniques)) => {
                    ess.push(new_ess);
                    uniques.push(new_uniques);
                }
                Err(err) => return Err(err),
            }
        }

        Ok((ess, uniques))
    }
}
