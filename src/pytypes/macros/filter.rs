/// A macro that generates a function that creates a new custom filter.
#[macro_export]
macro_rules! py_add_model_filter {
    ($model: ty, $name: ident, $nparams: expr, $filter_name: literal, $conf_type: ident, $conf_ndim: expr, $obs_ndim: expr, $model_ndim: expr) => {
        paste::paste! {
            #[derive(Clone)]
            #[doc = "The " $filter_name " filter for the " $name " model."]
            #[pyo3::pyclass(from_py_object)]
            pub struct [< $name $filter_name Filter >] (
                pub bayesfm::methods::filters::ParticleFilter<Float, bayesfm::conf::$conf_type<Float, $conf_ndim>, bayesfm::obs::ObsVec<Float, $obs_ndim>, $model<Float, prodef::MultivariateDensity<Float, nalgebra::Const<$nparams>>>, $model_ndim, $nparams>,
            );

            #[pyo3::pymethods]
            impl $name {
                #[pyo3(signature = (initial, configuration, ref_data, **opt_kwargs))]
                /// Create a new filter object from a model.
                pub fn [<new_ $filter_name:lower _filter>]<'py>(
                    &self,
                    py: pyo3::Python<'py>,
                    initial: bayesfm::pytypes:: [< Py $conf_type $conf_ndim>],
                    configuration: bayesfm::pytypes:: [< Py $conf_type $conf_ndim Series>],
                    ref_data: numpy::PyReadonlyArray2<Float>,
                    opt_kwargs: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>
                ) -> pyo3::PyResult<[<$name $filter_name Filter>]>
                {
                    // Exctract initial seed and size.
                    let (initial_seed, size) = match opt_kwargs {
                        Some(kwargs)  => {
                            let initial_seed = match pyo3::types::PyDictMethods::get_item(kwargs, "initial_seed")? {
                                Some(value) => pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None => 42,
                            };
                            let size = match pyo3::types::PyDictMethods::get_item(kwargs, "size")? {
                                Some(value) => pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None => 1024,
                            };

                            (initial_seed, size)
                        },
                        None => (42, 1024),
                    };

                    // Convert ref_data to a vector
                    let matrix = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Const<$obs_ndim>, nalgebra::Dyn>(ref_data, "ref_data")?;
                    let ref_vector = nalgebra::DVector::from_iterator(matrix.ncols(), matrix.column_iter().map(|col| bayesfm::obs::ObsVec::from(col.clone_owned())));

                    let obs_ensbl = match bayesfm::EnsembleObservations::new(initial.0, configuration.0, size, Some(ref_vector)) {
                        Some(obs_ensbl) => obs_ensbl,
                        None => return Err(pyo3::exceptions::PyRuntimeError::new_err("length of the configuration series does not match the length of ref_data")),
                    };

                    let ensbl = bayesfm::EnsembleState::new_zeros(size);

                    let mut fobj = py.detach(|| {
                        bayesfm::methods::filters::ParticleFilter::new(self.0.clone(), ensbl, obs_ensbl, initial_seed, None)
                    });

                    match opt_kwargs {
                        Some(kwargs) => {
                            match pyo3::types::PyDictMethods::get_item(kwargs, "exploration_factor")? {
                                Some(value) => fobj.settings.exploration_factor = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "max_iterations")? {
                                Some(value) => fobj.settings.max_iterations = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "effective_particle_threshold_factor")? {
                                Some(value) => fobj.settings.effective_particle_threshold_factor = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "simulation_ensemble_size_factor")? {
                                Some(value) => fobj.settings.simulation_ensemble_size_factor = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "simulation_time_limit")? {
                                Some(value) => fobj.settings.simulation_time_limit = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "simulation_time_prediction")? {
                                Some(value) => fobj.settings.simulation_time_prediction = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                        },
                        _ => ()
                    }

                    Ok([<$name $filter_name Filter>](fobj))
                }

                #[pyo3(signature = (initial, configuration, ref_data, input, **opt_kwargs))]
                /// Create a new filter object from a model and input ensemble.
                pub fn [<new_ $filter_name:lower _filter_with_particles>]<'py>(
                    &self,
                    py: pyo3::Python<'py>,
                    initial: bayesfm::pytypes:: [< Py $conf_type $conf_ndim>],
                    configuration: bayesfm::pytypes:: [< Py $conf_type $conf_ndim Series>],
                    ref_data: numpy::PyReadonlyArray2<Float>,
                    input: numpy::PyReadonlyArray2<Float>,
                    opt_kwargs: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>,
                ) -> pyo3::PyResult<[<$name $filter_name Filter>]> {
                    // Exctract initial seed and size.
                    let initial_seed = match opt_kwargs {
                        Some(kwargs)  => {
                            match pyo3::types::PyDictMethods::get_item(kwargs, "initial_seed")? {
                                Some(value) => pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None => 42,
                            }
                        },
                        None => 42,
                    };

                    // Convert ref_data to a vector
                    let matrix = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Const<$obs_ndim>, nalgebra::Dyn>(ref_data, "ref_data")?;
                    let ref_vector = nalgebra::DVector::from_iterator(matrix.ncols(), matrix.column_iter().map(|col| bayesfm::obs::ObsVec::from(col.clone_owned())));

                    let matrix = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Const<$nparams>, nalgebra::Dyn>(input, "input")?;

                    let ensbl = bayesfm::EnsembleState::from_matrix(matrix);

                    let obs_ensbl = match bayesfm::EnsembleObservations::new(initial.0, configuration.0, ensbl.len(), Some(ref_vector)) {
                        Some(obs_ensbl) => obs_ensbl,
                        None => return Err(pyo3::exceptions::PyRuntimeError::new_err("length of the configuration series does not match the length of ref_data")),
                    };

                    let mut fobj = py.detach(|| {
                        bayesfm::methods::filters::ParticleFilter::new(self.0.clone(), ensbl, obs_ensbl, initial_seed, None)
                    });

                    match opt_kwargs {
                        Some(kwargs) => {
                            match pyo3::types::PyDictMethods::get_item(kwargs, "exploration_factor")? {
                                Some(value) => fobj.settings.exploration_factor = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "max_iterations")? {
                                Some(value) => fobj.settings.max_iterations = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "effective_particle_threshold_factor")? {
                                Some(value) => fobj.settings.effective_particle_threshold_factor = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "simulation_ensemble_size_factor")? {
                                Some(value) => fobj.settings.simulation_ensemble_size_factor = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "simulation_time_limit")? {
                                Some(value) => fobj.settings.simulation_time_limit = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                            match pyo3::types::PyDictMethods::get_item(kwargs, "simulation_time_prediction")? {
                                Some(value) => fobj.settings.simulation_time_prediction = pyo3::FromPyObject::extract(value.as_borrowed())?,
                                None =>(),
                            }
                        },
                        _ => ()
                    };

                    Ok([<$name $filter_name Filter>](fobj))
                }
            }

            #[pyo3::pymethods]
            impl [<$name $filter_name Filter>] {
                /// Approximate Bayesian Computation iteration with Multivariate Normal Kernel.
                pub fn [< abc_mvnk_ $filter_name:lower>]<'py>(&mut self, py: pyo3::Python<'py>, metric: String, threshold: Float, rng: &mut bayesfm::pytypes::PyXoshiro256PlusPlus, noise: &mut bayesfm::pytypes::PyObsVecNoise) -> pyo3::PyResult<Float> {
                    let error = bayesfm::py_select_error_metric!(metric, bayesfm::obs::ObsVec<Float, $obs_ndim>);

                    py.detach(|| bayesfm::py_unroll_filter_errors!(self.0.abc_mvnk((&error, threshold), &$model::[< observe_ $filter_name:lower >], None, &mut rng.0, &mut noise.0)))
                }

                /// Blocked approximate Bayesian Computation iteration with Multivariate Normal Kernel.
                pub fn [< abc_mvnk_block_ $filter_name:lower>]<'py>(&mut self, py: pyo3::Python<'py>, metric: String, threshold: Float, dims: Vec<usize>, rng: &mut bayesfm::pytypes::PyXoshiro256PlusPlus, noise: &mut bayesfm::pytypes::PyObsVecNoise) -> pyo3::PyResult<Float> {
                    let error = bayesfm::py_select_error_metric!(metric, bayesfm::obs::ObsVec<Float, $obs_ndim>);

                    py.detach(|| bayesfm::py_unroll_filter_errors!(self.0.abc_mvnk((&error, threshold), &$model::[< observe_ $filter_name:lower >], Some(&dims), &mut rng.0, &mut noise.0)))
                }

                 /// Create a copy of the filter object.
                pub fn copy(&self) -> Self {
                    Self(self.0.clone())
                }

                 /// Differential evolution step.
                pub fn [< dev_ $filter_name:lower >]<'py>(&mut self, py: pyo3::Python<'py>, metric: String, mutation_factor: Float, recombination_factor: Float) -> pyo3::PyResult<usize> {
                    let error = bayesfm::py_select_error_metric!(metric, bayesfm::obs::ObsVec<Float, $obs_ndim>);

                    py.detach(|| bayesfm::py_unroll_model_errors!(self.0.dev(&error, &$model::[< observe_ $filter_name:lower >],(mutation_factor, recombination_factor))))
                }

                /// Return the errors of the current observation ensemble.
                pub fn errors(&self) -> Vec<Float> {
                    self.0.errors().iter().cloned().collect()
                }

                /// Return the error quantile of the current observation ensemble.
                pub fn error_quantile(&self, value: f64) -> Float {
                    self.0.error_quantile(value).unwrap()
                }

                /// Initialize the filter with a given error metric and threshold.
                #[pyo3(signature = (rng, metric="all".to_string(), threshold=1.0) )]
                pub fn [< initialize_ $filter_name:lower >]<'py>(&mut self, py: pyo3::Python<'py>, rng: &mut bayesfm::pytypes::PyXoshiro256PlusPlus, metric: String, threshold: Float) -> pyo3::PyResult<()> {
                    let error = bayesfm::py_select_error_metric!(metric, bayesfm::obs::ObsVec<Float, $obs_ndim>);
                    let filter = |o1: &[bayesfm::obs::ObsVec<Float, $obs_ndim>], o2: &[bayesfm::obs::ObsVec<Float, $obs_ndim>]| (error(o1, o2) < threshold, error(o1, o2));

                    py.detach(|| {
                        let prior = self.0.prior().clone();
                        bayesfm::py_unroll_filter_errors!(self.0.initialize(&filter,  &$model::[<observe_ $filter_name:lower >], prior, &mut rng.0))
                    })
                }

                /// Return the likelihoods of the current observation ensemble given a covariance matrix.
                pub fn likelihoods<'py>(&self, py: pyo3::Python<'py>, covariance: numpy::PyReadonlyArray2<Float>) -> pyo3::PyResult<Vec<Float>> {
                    let matrix = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Dyn, nalgebra::Dyn>(covariance, "covariance")?;
                    let params = matrix.nrows();

                    let mvnpdf = prodef::MultivariateNormalDensity::new(matrix, prodef::Domain::new_udomain(nalgebra::Dyn(params)), None).unwrap();

                    let llh = |o1: &[bayesfm::obs::ObsVec<Float, $obs_ndim>], o2: &[bayesfm::obs::ObsVec<Float, $obs_ndim>]| {
                        let mut value = bayesfm::obs::ov_error(o1, o2, bayesfm::obs::ObsVecMetric::Valid);

                        if value == 0.0 {
                            for i in 0..3 {
                                let veca = nalgebra::DVector::from_iterator(params, o1.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));

                                let vecb = nalgebra::DVector::from_iterator(params, o2.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));

                                let delta = vecb - veca;

                                value -= mvnpdf.mahalanobis_distance_sq::<nalgebra::U1, nalgebra::Dyn>(&delta.as_view());
                            }
                        }

                        value
                    };

                    py.detach(|| {
                        Ok(self.0.errors_func(&llh))
                    })
                }

                /// Return the covariance matrix of the particle filter kernel.
                pub fn mvnk<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<Float>> {
                    let mvnk: prodef::MultivariateNormalDensity<Float, nalgebra::Const<$nparams>> =
                        prodef::MultivariateNormalDensity::from_vectors::<nalgebra::U1, nalgebra::Const<$nparams>>(
                            &self.0.particles().as_view(),
                            prodef::Domain::new_udomain(nalgebra::Const::<$nparams>),
                            self.0.weights().map(|w| w.as_slice()),
                        )
                        .unwrap();

                    numpy::ToPyArray::to_pyarray(&mvnk.covariance_matrix().clone_owned(), py)
                }

                /// Return the covariance matrix of the particle filter kernel.
                pub fn mvnk_ltm<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<Float>> {
                    let mvnk: prodef::MultivariateNormalDensity<Float, nalgebra::Const<$nparams>> =
                        prodef::MultivariateNormalDensity::from_vectors::<nalgebra::U1, nalgebra::Const<$nparams>>(
                            &self.0.particles().as_view(),
                            prodef::Domain::new_udomain(nalgebra::Const::<$nparams>),
                            self.0.weights().map(|w| w.as_slice()),
                        )
                        .unwrap();

                    numpy::ToPyArray::to_pyarray(&mvnk.lower_triangular_matrix().transpose().clone_owned(), py)
                }

                /// Return the particles and their weights as numpy arrays.
                pub fn particles<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Bound<'py, numpy::PyArray2<Float>>, Option<Vec<Float>>) {
                    (
                        numpy::ToPyArray::to_pyarray(&self.0.particles().transpose().clone_owned(), py),
                        self.0.weights().as_ref().map(|w| w.iter().cloned().collect::<Vec<Float>>())
                    )
                }

                /// Simulate observations and return the observation ensemble and errors.
                #[pyo3(signature = (opt_configuration = None))]
                pub fn simulate(
                    &mut self,
                    opt_configuration: Option<bayesfm::pytypes:: [< Py $conf_type $conf_ndim Series>]>,
                ) -> pyo3::PyResult<bayesfm::pytypes:: [<PyEnsbl $conf_type $conf_ndim ObsVec $obs_ndim>]> {
                    let obs_ensbl = match opt_configuration {
                        Some(configuration) => {
                            bayesfm::py_unroll_filter_errors!(self.0.simulate(Some(&configuration.0), None, &$model::[< observe_ $filter_name:lower >], None::<(&bayesfm::noise::NullNoise, &mut rand::rngs::Xoshiro256PlusPlus)>))

                        },
                        _ => bayesfm::py_unroll_filter_errors!(self.0.simulate(None, None, &$model::[< observe_ $filter_name:lower >], None::<(&bayesfm::noise::NullNoise, &mut rand::rngs::Xoshiro256PlusPlus)>)),

                    }?;

                    Ok(obs_ensbl.into())
                }

                /// Simulate observations and return the observation ensemble and errors.
                #[pyo3(signature = (metric="rmse".to_string(), opt_configuration = None, opt_ref_data = None))]
                pub fn simulate_with_errors<'py>(
                    &mut self,
                    py: pyo3::Python<'py>,
                    metric: String,
                    opt_configuration: Option<bayesfm::pytypes:: [< Py $conf_type $conf_ndim Series>]>,
                    opt_ref_data: Option<numpy::PyReadonlyArray2<Float>>
                ) -> pyo3::PyResult<(bayesfm::pytypes:: [<PyEnsbl $conf_type $conf_ndim ObsVec $obs_ndim>], Vec<Float>)> {
                    let error = bayesfm::py_select_error_metric!(metric, bayesfm::obs::ObsVec<Float, $obs_ndim>);

                    let obs_ensbl = match (opt_configuration, opt_ref_data) {
                        (Some(configuration), Some(ref_data)) => {
                            let iter = bayesfm::py_any_iterator!(ref_data, [Float; $obs_ndim]).map(bayesfm::obs::ObsVec::from);

                            bayesfm::py_unroll_filter_errors!(self.0.simulate(Some(&configuration.0), Some(&nalgebra::DVector::from(Vec::from_iter(iter))), &$model::[< observe_ $filter_name:lower >], None::<(&bayesfm::noise::NullNoise, &mut rand::rngs::Xoshiro256PlusPlus)>))

                        },
                        (None, None) => bayesfm::py_unroll_filter_errors!(self.0.simulate(None, None, &$model::[< observe_ $filter_name:lower >], None::<(&bayesfm::noise::NullNoise, &mut rand::rngs::Xoshiro256PlusPlus)>)),
                        _ => Err(pyo3::exceptions::PyValueError::new_err("a new observation must be combined with a new reference data"))?,
                    }?;

                    py.detach(|| {
                        let errors = obs_ensbl.errors_func(&error);

                        Ok((obs_ensbl.into(), errors))
                    })
                }

                /// Sequential importance resampling iteration with a multivariate Normal Kernel and a covariance matrix.
                pub fn [< sir_mvnk_ $filter_name:lower>]<'py>(&mut self, py: pyo3::Python<'py>, covariance: numpy::PyReadonlyArray2<Float>, rng: &mut bayesfm::pytypes::PyXoshiro256PlusPlus) -> pyo3::PyResult<(Float, usize)> {
                    let matrix = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Dyn, nalgebra::Dyn>(covariance, "covariance")?;
                    let obscount = matrix.nrows();

                    let mvnpdf = prodef::MultivariateNormalDensity::new(matrix, prodef::Domain::new_udomain(nalgebra::Dyn(obscount)), None).unwrap();

                    let llh = |o1: &[bayesfm::obs::ObsVec<Float, $obs_ndim>], o2: &[bayesfm::obs::ObsVec<Float, $obs_ndim>]| {
                        let mut value = bayesfm::obs::ov_error(o1, o2, bayesfm::obs::ObsVecMetric::Valid);

                        if value == 0.0 {
                            for i in 0..3 {
                                let veca = nalgebra::DVector::from_iterator(obscount, o1.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));

                                let vecb = nalgebra::DVector::from_iterator(obscount, o2.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));

                                let delta = vecb - veca;

                                value -= mvnpdf.mahalanobis_distance_sq::<nalgebra::U1, nalgebra::Dyn>(&delta.as_view());
                            }
                        }

                        value
                    };

                    py.detach(|| bayesfm::py_unroll_filter_errors!(self.0.sir_mvnk(&$model::[< observe_ $filter_name:lower >], &llh, &mut rng.0)))
                }

                /// Sequential importance resampling iteration with a multivariate Normal Kernel and a fixed percentage error.
                pub fn [< sir_mvnk_perc_ $filter_name:lower>]<'py>(&mut self, py: pyo3::Python<'py>, errors: Vec<Float>, rng: &mut bayesfm::pytypes::PyXoshiro256PlusPlus) -> pyo3::PyResult<(Float, usize)> {
                    let llh = |o1: &[bayesfm::obs::ObsVec<Float, $obs_ndim>], o2: &[bayesfm::obs::ObsVec<Float, $obs_ndim>]| {
                        let mut value = bayesfm::obs::ov_error(o1, o2, bayesfm::obs::ObsVecMetric::Valid);

                        if value == 0.0 {
                            for i in 0..$obs_ndim {
                                let veca = nalgebra::DVector::from_iterator(errors.len(), o1.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));

                                let vecb = nalgebra::DVector::from_iterator(errors.len(), o2.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));

                                value -= veca.iter().zip(vecb.iter()).zip(errors.iter()).map(|((y, f), e)| (y - f).powi(2) / f.powi(2) / 2.0 / e.powi(2)).sum::<Float>()
                            }
                        }

                        value
                    };

                    py.detach(|| bayesfm::py_unroll_filter_errors!(self.0.sir_mvnk(&$model::[< observe_ $filter_name:lower >], &llh, &mut rng.0)))
                }

                /// Return the size of the filter object ensemble.
                pub fn size(&self) -> usize {
                    self.0.len()
                }
            }
        }
    };
}

pub use py_add_model_filter;
