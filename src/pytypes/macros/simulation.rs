/// A macro that generates a function to simulate an observable.
#[macro_export]
macro_rules! py_add_model_simulation {
    ($model: ty, $name: ident, $nparams: expr, $observable: expr, $conf_type: ident, $conf_ndim: expr, $obs_ndim: expr) => {
        paste::paste! {
            #[pyo3::pymethods]
            impl $name
            {
                #[doc = "Compute the fisher information matrix for the" $observable "for a configuration time-series and a specific set of model parameters with a given covariance."]
                pub fn [< fisher_ $observable>]<'py>(
                    &self,
                    py: pyo3::Python<'py>,
                    initial: bayesfm::pytypes:: [< Py $conf_type $conf_ndim>],
                    configuration: bayesfm::pytypes:: [<Py $conf_type $conf_ndim Series>],
                    input: numpy::PyReadonlyArray2<Float>,
                    covariance: numpy::PyReadonlyArray2<Float>
                ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray2<Float>>> {
                    let params = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Const<$nparams>, nalgebra::Dyn>(input, "input")?;
                    let matrix = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Dyn, nalgebra::Dyn>(covariance, "covariance")?;

                    let mvnk = prodef::MultivariateNormalDensity::new(matrix.clone(), prodef::Domain::new_udomain(nalgebra::Dyn(matrix.nrows())), None).unwrap();

                    let fisher = py.detach(|| {
                        let fim = $crate::methods::fisher_information_matrix(
                            &self.0, (&initial.0, &configuration.0),
                            &params.column(0),
                            &$model::[< observe_ $observable:lower >],
                            &mvnk);

                        Ok::<_, pyo3::PyErr>(fim.unwrap())
                    })?;

                    Ok(fisher.transpose().to_pyarray(py))
                }

                #[doc = "Simulate the" $observable "for a configuration time-series and input array."]
                #[pyo3(signature = (initial, configuration, input, rng, opt_noise = None))]
                pub fn [< simulate_ $observable >]<'py>(
                    &self,
                    py: pyo3::Python<'py>,
                    initial: bayesfm::pytypes:: [< Py $conf_type $conf_ndim>],
                    configuration: bayesfm::pytypes:: [<Py $conf_type $conf_ndim Series>],
                    input: numpy::PyReadonlyArray2<Float>,
                    rng: &mut bayesfm::pytypes::PyXoshiro256PlusPlus,
                    opt_noise: Option<bayesfm::pytypes::PyObsVecNoise>
                ) -> pyo3::PyResult<bayesfm::pytypes:: [<PyEnsbl $conf_type $conf_ndim ObsVec $obs_ndim>]> {
                    let matrix = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Const<$nparams>, nalgebra::Dyn>(input, "input")?;

                    py.detach(|| {
                        let mut obs_ensbl = bayesfm::EnsembleObservations::new(initial.0, configuration.0, matrix.ncols(), None).unwrap();

                        let mut ensbl = bayesfm::EnsembleState::new(matrix.clone_owned(), None, None);
                        bayesfm::py_unroll_model_errors!(ensbl.initialize(&self.0))?;

                        match opt_noise {
                            Some(noise) => bayesfm::py_unroll_model_errors!(bayesfm::EnsembleModel::simulate_ensbl_par(&self.0, &mut ensbl, &mut obs_ensbl, &$model::[<observe_ $observable >], Some((&noise.0.clone(), &mut rng.0))))?,
                            None => bayesfm::py_unroll_model_errors!(bayesfm::EnsembleModel::simulate_ensbl_par(&self.0, &mut ensbl, &mut obs_ensbl, &$model::[<observe_ $observable >], None::<(&bayesfm::noise::NullNoise, &mut rand::rngs::Xoshiro256PlusPlus)>))?
                        }

                        Ok(obs_ensbl.clone().into())
                    })
                }
            }
        }
    };
}

pub use py_add_model_simulation;
