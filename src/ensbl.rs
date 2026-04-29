use crate::{Model, ModelError, conf::ConfSeries, noise::Noise, obs::Obs};
use log::debug;
use nalgebra::{
    Const, DMatrix, DVector, DVectorView, Dyn, Matrix, MatrixView, MatrixViewMut, OMatrix,
    RealField, SVectorView, Scalar, U1, VecStorage, VectorView,
};
use num_traits::Zero;
use prodef::{Density, ParticleDensity, SamplingMode};
use rand::SeedableRng;
use rand_distr::uniform::SampleUniform;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, iter::Sum, ops::Sub, time::Instant};

/// A trait that is shared by all Bayesian forward ensemble models within the crate's framework.
///
/// # Type Parameters
/// - `T`: Numeric field type (e.g., f32 or f64 implementing `RealField`)
/// - `D`: Number of spatial dimensions (const generic)
/// - `P`: Number of model parameters (const generic)
///
/// # Associated Constants
/// - `RAYON_CHUNK_SIZE`: Rayon chunk size (used for parallel operations to minimize overhead)
///
/// # Parallel Operations
/// This trait extends [`Model`] with ensemble-level operations that leverage Rayon for parallelism:
/// - Operates on `EnsembleState<T, CSST, FMST, D, P>` containing N ensemble members
/// - Uses deterministic RNG seeding per chunk: `Xoshiro256PlusPlus::seed_from_u64(rseed + chunk_id * MAGIC_CONST)`
/// - Chunks with `.chunks(Self::RAYON_CHUNK_SIZE)` to balance parallelism overhead
pub trait EnsembleModel<T, const D: usize, const P: usize>: Model<T, D, P>
where
    T: RealField,
    Self: Sync,
{
    /// The default rayon chunk size that is used for parallel iterators.
    ///
    /// Certain operations may use multiples, or fractions, of this value.
    const RAYON_CHUNK_SIZE: usize;

    /// Initialize the model parameters, and both the coordinate system and forward model states for an ensemble.
    fn initialize_ensbl<G>(
        &self,
        ensbl: &mut EnsembleState<T, Self::CSST, Self::FMST, D, P>,
        prior: G,
        mode: &SamplingMode,
        rseed: u64,
    ) -> Result<(), ModelError<T>>
    where
        G: Density<T, Const<P>> + Sync,
    {
        let start = Instant::now();

        assert!(
            !ensbl.is_empty(),
            "cannot initialize an empty model ensemble"
        );

        ensbl
            .params
            .par_column_iter_mut()
            .zip(ensbl.states.par_iter_mut())
            .chunks(Self::RAYON_CHUNK_SIZE)
            .enumerate()
            .try_for_each(|(chunk_id, mut chunk)| {
                let mut rng =
                    Xoshiro256PlusPlus::seed_from_u64(rseed + (chunk_id * 310248241) as u64);

                chunk
                    .iter_mut()
                    .try_for_each(|(params, (fm_state, cs_state))| {
                        self.initialize::<G>(
                            params,
                            fm_state,
                            cs_state,
                            prior.clone(),
                            mode,
                            &mut rng,
                        )?;

                        Ok::<(), ModelError<T>>(())
                    })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "initialize_ensbl: {:2.1}k evaluations in {:.0}ms",
            ensbl.len() as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }

    /// Initialize the model parameters for an ensemble.
    fn initialize_params_ensbl<G>(
        &self,
        ensbl: &mut EnsembleState<T, Self::CSST, Self::FMST, D, P>,
        prior: G,
        mode: &SamplingMode,
        rseed: u64,
    ) -> Result<(), ModelError<T>>
    where
        G: Density<T, Const<P>> + Sync,
    {
        let start = Instant::now();

        ensbl
            .params
            .par_column_iter_mut()
            .chunks(Self::RAYON_CHUNK_SIZE)
            .enumerate()
            .try_for_each(|(chunk_id, mut chunk)| {
                let mut rng =
                    Xoshiro256PlusPlus::seed_from_u64(rseed + (chunk_id * 213161503) as u64);

                chunk.iter_mut().try_for_each(|params| {
                    self.initialize_params(params, prior.clone(), mode, &mut rng)?;

                    Ok::<(), ModelError<T>>(())
                })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "initialize_params_ensbl: {:2.1}k evaluations in {:.0}ms",
            ensbl.len() as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }

    /// Initialize the the coordinate system and forward model states for an ensemble.
    fn initialize_states_ensbl(
        &self,
        ensbl: &mut EnsembleState<T, Self::CSST, Self::FMST, D, P>,
    ) -> Result<(), ModelError<T>>
where {
        let start = Instant::now();

        ensbl
            .params
            .par_column_iter()
            .zip(ensbl.states.par_iter_mut())
            .chunks(Self::RAYON_CHUNK_SIZE)
            .try_for_each(|mut chunk| {
                chunk
                    .iter_mut()
                    .try_for_each(|(params, (fm_state, cs_state))| {
                        self.initialize_states(params, fm_state, cs_state)?;

                        Ok::<(), ModelError<T>>(())
                    })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "initialize_states_ensbl: {:2.1}k evaluations in {:.0}ms",
            ensbl.len() as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }

    /// Perform an ensemble forward simulation and generate synthetic observables `OD` for the
    /// given spacecraft observers for a given generating function `OF` and noise model `NM`.
    fn simulate_ensbl<OC, OD, OF, NM>(
        &self,
        ensbl: &mut EnsembleState<T, Self::CSST, Self::FMST, D, P>,
        obs_ensbl: &mut EnsembleObservations<OC, OD>,
        obs_func: &OF,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OC: Scalar,
        for<'a> &'a OC: Sub<&'a OC, Output = T>,
        OD: Obs,
        OF: Fn(
                &Self,
                &OC,
                &SVectorView<T, P>,
                &Self::FMST,
                &Self::CSST,
            ) -> Result<OD, ModelError<T>>
            + Sync,
        NM: Noise<OD>,
    {
        let start = Instant::now();

        let mut last_observation = &obs_ensbl.initial().clone();

        obs_ensbl
            .time_iter_mut()
            .try_for_each(|(conf, mut obs_row)| {
                // Compute time step to next configuration.
                let time_step = conf - last_observation;
                last_observation = conf;

                if time_step < T::zero() {
                    return Err(ModelError::Evolution(time_step));
                } else {
                    ensbl
                        .params
                        .column_iter()
                        .zip(ensbl.states.iter_mut())
                        .zip(obs_row.column_iter_mut())
                        .try_for_each(|((params, (fm_state, cs_state)), mut obs)| {
                            self.evolve_fmst(time_step.clone(), &params, fm_state, cs_state)?;

                            obs[(0, 0)] = obs_func(self, conf, &params, fm_state, cs_state)?;

                            Ok::<(), ModelError<T>>(())
                        })?;
                }

                Ok::<(), ModelError<T>>(())
            })?;

        if let Some(noise) = opt_noise {
            let mut rng = noise.initialize_rng(37, 23);

            obs_ensbl.ensbl_iter_mut().for_each(|(_, mut col)| {
                noise.add_noise(&mut col, &mut rng);
            });

            noise.increment_random_seed()
        }

        debug!(
            "simulate_ensbl: {:2.1}k evaluations in {:.0}ms",
            (obs_ensbl.len() * ensbl.len()) as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }

    /// Perform an ensemble forward simulation, in parallel, and generate synthetic observables `OD` for the
    /// given spacecraft observers for a given generating function `OF` and noise model `NM`.
    fn simulate_ensbl_par<OC, OD, OF, NM>(
        &self,
        ensbl: &mut EnsembleState<T, Self::CSST, Self::FMST, D, P>,
        obs_ensbl: &mut EnsembleObservations<OC, OD>,
        obs_func: &OF,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OC: Scalar + Sync,
        for<'a> &'a OC: Sub<&'a OC, Output = T>,
        OD: Obs,
        OF: Fn(
                &Self,
                &OC,
                &SVectorView<T, P>,
                &Self::FMST,
                &Self::CSST,
            ) -> Result<OD, ModelError<T>>
            + Sync,
        NM: Noise<OD> + Sync,
    {
        let start = Instant::now();

        let mut last_observation = &obs_ensbl.initial().clone();

        obs_ensbl
            .time_iter_mut()
            .try_for_each(|(conf, mut obs_row)| {
                // Compute time step to next configuration.
                let time_step = conf - last_observation;
                last_observation = conf;

                if time_step < T::zero() {
                    return Err(ModelError::Evolution(time_step));
                } else {
                    ensbl
                        .params
                        .par_column_iter()
                        .zip(ensbl.states.par_iter_mut())
                        .zip(obs_row.par_column_iter_mut())
                        .chunks(Self::RAYON_CHUNK_SIZE)
                        .try_for_each(|mut chunk| {
                            chunk.iter_mut().try_for_each(
                                |((params, (fm_state, cs_state)), obs)| {
                                    self.evolve_fmst(
                                        time_step.clone(),
                                        params,
                                        fm_state,
                                        cs_state,
                                    )?;

                                    obs[(0, 0)] = obs_func(self, conf, params, fm_state, cs_state)?;

                                    Ok::<(), ModelError<T>>(())
                                },
                            )?;

                            Ok::<(), ModelError<T>>(())
                        })?;
                }

                Ok::<(), ModelError<T>>(())
            })?;

        if let Some(noise) = opt_noise {
            obs_ensbl
                .par_ensbl_iter_mut()
                .chunks(Self::RAYON_CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk_id, mut chunk)| {
                    let mut rng = noise.initialize_rng(29 * chunk_id as u64, 23);

                    chunk.iter_mut().for_each(|(_, col)| {
                        noise.add_noise(col, &mut rng);
                    });
                });

            noise.increment_random_seed()
        }

        debug!(
            "simulate_ensbl_par: {:2.1}k evaluations in {:.0}ms",
            (obs_ensbl.len() * ensbl.len()) as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }
}

/// Manages observation outputs across N ensemble members and M time steps.
///
/// This structure pairs observation configurations (times) with observation data
/// from each ensemble member's forward simulation. It also optionally stores
/// reference observations for comparison and error calculation.
///
/// # Generic Parameters
/// - `OC`: Observer configuration type (typically `DateTime<Utc>` or scalar timestamp)
/// - `OD`: Observation data type (must implement [`Obs`], e.g., `ObsVec<T, N>`, `f64`)
///
/// # Error Calculation
/// Use [`errors_func`](Self::errors_func) to compute error metrics between
/// ensemble members and reference data, or
/// [`errors_with_threshold`](Self::errors_with_threshold) to threshold and
/// filter by acceptance criteria.
///
/// # Examples
/// ```ignore
/// use bayesfm::{EnsembleObservations, ObsVec, ConfSeries};
///
/// // Create observation times for 3 timesteps
/// let config = ConfSeries::from_iter(vec![0.0, 1.0, 2.0]);
///
/// // Initialize for 32 ensemble members with optional reference data
/// let mut obs = EnsembleObservations::new(0.0, config, 32, None)?;
///
/// // Access observations for ensemble member 5 at all times
/// let member_5_obs = obs.output(5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EnsembleObservations<OC, OD>
where
    OC: Scalar,
    OD: Scalar,
{
    /// Initial observation configuration (reference for computing time-steps).
    initial: OC,

    /// The underlying configuration time-series.
    configuration: ConfSeries<OC>,

    /// Obs ensemble array.
    observations: DMatrix<OD>,

    /// Obs reference array (optional).
    ref_output: Option<DVector<OD>>,
}

impl<OC, OD> EnsembleObservations<OC, OD>
where
    OC: Scalar,
    OD: Scalar + Zero,
{
    /// Iterate over the ensemble.
    pub fn ensbl_iter(
        &self,
    ) -> impl Iterator<Item = (&ConfSeries<OC>, MatrixView<'_, OD, Dyn, U1>)> {
        self.observations
            .column_iter()
            .map(|col| (&self.configuration, col))
    }

    /// Mutably iterate over the ensemble.
    pub fn ensbl_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (&ConfSeries<OC>, MatrixViewMut<'_, OD, Dyn, U1>)> {
        self.observations
            .column_iter_mut()
            .map(|col| (&self.configuration, col))
    }

    /// Return a list of error values for a given error metric.
    pub fn errors_func<T, EF>(&self, func: &EF) -> Vec<T>
    where
        T: Send + Sync,
        OC: Sync,
        OD: Send + Sync,
        EF: Fn(&[OD], &[OD]) -> T + Sync,
    {
        self.par_ensbl_iter()
            .map(|(_, out)| {
                func(
                    self.ref_output
                        .as_ref()
                        .expect("reference data missing")
                        .as_slice(),
                    out.as_slice(),
                )
            })
            .collect::<Vec<T>>()
    }

    /// Return a list of error values and flags using a threshold value for a given error metric.
    pub fn errors_with_threshold<T, EF>(&self, func: &EF, threshold: T) -> (Vec<T>, Vec<bool>)
    where
        T: PartialOrd + Send + Sync,
        OC: Sync,
        OD: Send + Sync,
        EF: Fn(&[OD], &[OD]) -> T + Sync,
    {
        let mut flags = vec![true; self.observations.ncols()];

        let values = self
            .par_ensbl_iter()
            .zip(flags.par_iter_mut())
            .chunks(128)
            .map(|mut chunk| {
                chunk
                    .iter_mut()
                    .map(|((_, out), flag)| {
                        let value = func(
                            self.ref_output
                                .as_ref()
                                .expect("reference data missing")
                                .as_slice(),
                            out.as_slice(),
                        );

                        **flag = value < threshold;

                        value
                    })
                    .collect::<Vec<T>>()
            })
            .flatten()
            .collect::<Vec<T>>();

        (values, flags)
    }

    /// Return a reference to an individual output column.
    pub fn output(&self, index: usize) -> DVectorView<'_, OD> {
        assert!(
            index < self.observations.ncols(),
            "cannot get output, index out of bounds"
        );

        self.observations.column(index)
    }

    /// Return a reference to multiple output columns.
    pub fn outputs(&self, indices: &[usize]) -> Vec<DVectorView<'_, OD>> {
        indices.iter().map(|&index| self.output(index)).collect()
    }

    /// Return a reference to the initial configuration configuration.
    pub fn initial(&self) -> &OC {
        &self.initial
    }

    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the configuration.
    pub fn len(&self) -> usize {
        self.configuration.len()
    }

    /// Create a new [`crate::configuration::data::ObsData`].
    pub fn new(
        initial: OC,
        configuration: ConfSeries<OC>,
        size: usize,
        opt_ref_data: Option<DVector<OD>>,
    ) -> Option<Self> {
        let slen = configuration.len();

        match opt_ref_data.as_ref() {
            Some(data) if data.len() != slen => None,
            _ => Some(Self {
                observations: DMatrix::<OD>::zeros(slen, size),
                ref_output: opt_ref_data,
                configuration,
                initial,
            }),
        }
    }

    /// Iterate over the ensemble in parallel.
    pub fn par_ensbl_iter(
        &self,
    ) -> impl IndexedParallelIterator<Item = (&ConfSeries<OC>, MatrixView<'_, OD, Dyn, U1>)>
    where
        OC: Sync,
        OD: Send + Sync,
    {
        self.observations
            .par_column_iter()
            .map(|col| (&self.configuration, col))
    }

    /// Mutably iterate over the ensemble in parallel.
    pub fn par_ensbl_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = (&ConfSeries<OC>, MatrixViewMut<'_, OD, Dyn, U1>)>
    where
        OC: Sync,
        OD: Send + Sync,
    {
        self.observations
            .par_column_iter_mut()
            .map(|col| (&self.configuration, col))
    }

    /// Returns the reference observations, of the internal [`ConfSeries`], as a slice.
    pub fn ref_output(&self) -> Option<&DVector<OD>> {
        self.ref_output.as_ref()
    }

    /// Return a reference to the internal [`ConfSeries`]
    pub fn configuration(&self) -> &ConfSeries<OC> {
        &self.configuration
    }

    /// Set an individual output column to the given value.
    pub fn set_output(&mut self, index: usize, column: &DVectorView<OD>) {
        assert!(
            index < self.observations.ncols(),
            "cannot set output, index out of bounds"
        );

        self.observations.set_column(index, column)
    }

    /// Return the size of the ensemble.
    pub fn size(&self) -> usize {
        self.observations.ncols()
    }

    /// Iterate over the ensemble along the time axis.
    pub fn time_iter(
        &mut self,
    ) -> impl Iterator<Item = (&OC, MatrixView<'_, OD, U1, Dyn, U1, Dyn>)> {
        (&self.configuration)
            .into_iter()
            .zip(self.observations.row_iter())
    }

    /// Mutably iterate over the ensemble along the time axis.
    pub fn time_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (&OC, MatrixViewMut<'_, OD, U1, Dyn, U1, Dyn>)> {
        (&self.configuration)
            .into_iter()
            .zip(self.observations.row_iter_mut())
    }
}

/// A object that holds an ensemble of initial parameters and states.
///
/// Optionally, also holds ensemble weights for the initial parameters
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "
    T: Serialize,
    FMST: Serialize,
    CSST: Serialize"))]
#[serde(bound(deserialize = "
    T: Deserialize<'de>, 
    FMST: Deserialize<'de>,
    CSST: Deserialize<'de>"))]
pub struct EnsembleState<T, CSST, FMST, const D: usize, const P: usize>
where
    T: RealField,
{
    /// Input parameters for each ensemble member.
    params: OMatrix<T, Const<P>, Dyn>,

    /// Coordinate and forward system states for each ensemble member.
    states: Vec<(FMST, CSST)>,

    /// Optional ensemble weights for the params parameters.
    opt_weights: Option<Vec<T>>,
}

impl<T, CSST, FMST, const P: usize, const D: usize> EnsembleState<T, CSST, FMST, D, P>
where
    T: RealField,
{
    /// Initialize the states of the ensemble.
    pub fn initialize<M>(&mut self, model: &M) -> Result<(), ModelError<T>>
    where
        M: EnsembleModel<T, D, P, CSST = CSST, FMST = FMST> + Sync,
    {
        model.initialize_states_ensbl(self)?;

        Ok(())
    }

    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of members in the ensemble.
    pub fn len(&self) -> usize {
        self.params.ncols()
    }

    /// Create a new [`EnsembleState`] from an existing parameter array.
    pub fn new(
        params: OMatrix<T, Const<P>, Dyn>,
        opt_states: Option<Vec<(FMST, CSST)>>,
        opt_weights: Option<Vec<T>>,
    ) -> Self
    where
        T: RealField + Sum,
        FMST: Clone + Default,
        CSST: Clone + Default,
    {
        let size = params.ncols();

        Self {
            params: params.clone_owned(),
            states: opt_states.unwrap_or(vec![(FMST::default(), CSST::default()); size]),
            opt_weights,
        }
    }

    /// Create a new [`EnsembleState`] with all parameters zero'd out.
    pub fn new_zeros(size: usize) -> Self
    where
        T: RealField + Sum,
        FMST: Clone + Default,
        CSST: Clone + Default,
    {
        let params = OMatrix::<T, Const<P>, Dyn>::zeros(size);
        let states = vec![(FMST::default(), CSST::default()); size];
        Self {
            params,
            states,
            opt_weights: None,
        }
    }

    /// Resample the model params parameters from a particle density, and re-initialize the states.
    pub fn new_resampled<M, G>(
        model: &M,
        size: usize,
        ptpdf: &ParticleDensity<T, Const<P>, G>,
        rseed: u64,
    ) -> Result<Self, ModelError<T>>
    where
        T: SampleUniform + Sum,
        CSST: Clone + Default + Send,
        FMST: Clone + Default + Send,
        M: EnsembleModel<T, D, P, CSST = CSST, FMST = FMST> + Sync,
        G: Sync,
    {
        let start = Instant::now();

        let mut obj = Self::new(
            Matrix::<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>::zeros(size),
            None,
            None,
        );

        obj.params
            .par_column_iter_mut()
            .zip(obj.states.par_iter_mut())
            .chunks(M::RAYON_CHUNK_SIZE)
            .enumerate()
            .try_for_each(|(chunk_id, mut chunk)| {
                let mut rng =
                    Xoshiro256PlusPlus::seed_from_u64(rseed + (chunk_id * 679389209) as u64);

                chunk
                    .iter_mut()
                    .try_for_each(|(params, (fm_state, cs_state))| {
                        params.set_column(0, &ptpdf.sample_particle(&mut rng));

                        model.initialize_states(&params.as_view(), fm_state, cs_state)?;

                        Ok::<(), ModelError<T>>(())
                    })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "ensemble_states new_resampled: {:2.3}M evaluations in {:.2} sec",
            obj.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        obj.opt_weights = None;

        Ok(obj)
    }

    /// Returns a reference to the optional ensemble weights.
    pub fn opt_weights(&self) -> Option<&Vec<T>> {
        self.opt_weights.as_ref()
    }

    /// Returns a reference to the underlying parameter matrix.
    pub fn params<'a>(&'a self) -> MatrixView<'a, T, Const<P>, Dyn> {
        self.params.as_view()
    }

    /// Returns a reference to the underlying parameter matrix.
    pub fn params_mut<'a>(&'a mut self) -> MatrixViewMut<'a, T, Const<P>, Dyn> {
        self.params.as_view_mut()
    }

    /// Set the optional ensemble weights.
    pub fn set_opt_weights(&mut self, new_weights: Option<Vec<T>>) {
        self.opt_weights = new_weights;
    }

    /// Sets an individual parameter column of the ensemble.
    pub fn set_param(&mut self, index: usize, params: &VectorView<T, Const<P>, U1>) {
        self.params.set_column(index, params)
    }

    /// Set the parameters of the ensemble.
    pub fn set_params(&mut self, params: &MatrixView<T, Const<P>, Dyn>) {
        self.params = params.clone_owned();
    }

    /// Serialize this data structure to a file using the JSON5 format.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json5::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }

    /// Returns a reference to the indexed state tuple.
    pub fn state(&self, index: usize) -> &(FMST, CSST) {
        &self.states[index]
    }
}
