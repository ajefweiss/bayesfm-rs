use crate::{
    EnsembleObservations,
    conf::{BasicConf, ConfSeries},
    obs::ObsVec,
    pytypes::*,
};
use nalgebra::{Const, DMatrix, DVector, DimSum, Dyn, SVector, U1};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray, ndarray::Dim};
use paste::paste;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

/// Export the PyEnsblBasicConfObsVec classes.
#[macro_export]
macro_rules! export_py_ensbl_basicconf_obsvec {
    ($module: expr, $(($ndim: expr, $nval: expr)),+) => {
        $(paste::paste!{
            $module.add_class::<[<PyEnsblBasicConf $ndim ObsVec $nval>]>()?;
        });+
    };
}

macro_rules! impl_py_ensbl_basicconf_obsvec {
    ($ndim: expr, $nval: expr) => {
        paste! {
            #[derive(Clone)]
            #[doc="PyEnsblVec for BasicConf(" $ndim ") and ObsVec("  $nval ")."]
            #[pyclass(from_py_object, name = EnsblBasicConf $ndim ObsVec $nval)]
            pub struct [<PyEnsblBasicConf $ndim ObsVec $nval>](pub EnsembleObservations<BasicConf<Float, $ndim>, ObsVec<Float, $nval>>);

            #[pymethods]
            impl [<PyEnsblBasicConf $ndim ObsVec $nval>] {
                /// Return the inner value of ensemble member(s) as a 2D numpy array.
                pub fn get<'py>(&self, py: Python<'py>, key: usize) -> PyResult<Bound<'py, PyArray2<Float>>> {
                    let matrix = if key < self.0.len() {
                        let column = self.0.output(key);

                        DMatrix::from_iterator(
                            $nval,
                            self.0.len(),
                            column.iter().flat_map(|value| value.iter().cloned()),
                        )
                    } else {
                        return Err(PyValueError::new_err("key must be smaller than ensemble size"));
                    };

                    Ok(matrix.transpose().to_pyarray(py))
                }

                /// Create a new PyEnsblBasicConf object from a BasicConfSeries.
                #[classmethod]
                #[pyo3(signature = (initial_timestamp, configuration, size = 1024, opt_ref_data = None))]
                pub fn from_conf(_cls: &Bound<PyType>, initial_timestamp: Float, configuration: [<PyBasicConf $ndim Series>], size: usize, opt_ref_data: Option<PyReadonlyArray2<Float>>) -> PyResult<Self>
                {
                    let ref_data = match opt_ref_data {
                        Some(ref_data) => {
                            // This array only contains values and not vectors.
                            let matrix = array_to_matrix::<Dim<[usize; 2]>, Const<$nval>, Dyn>(ref_data, "ref_data")?;

                           Some(DVector::from_iterator(matrix.ncols(), matrix.column_iter().map(|col| ObsVec::from(col.clone_owned()))))
                        },
                        None => None
                    };

                    let ensblobs = EnsembleObservations::new(
                        BasicConf::from((initial_timestamp, SVector::from_vec(vec![0.0; $ndim]))),
                        configuration.0.clone(),
                        size,
                        ref_data,
                    );

                    match ensblobs {
                        Some(ensblobs) => Ok(Self(ensblobs)),
                        None => Err(PyValueError::new_err(format!("configuration and optional reference data is mismatched in length"))),
                    }
                }

                #[new]
                #[pyo3(signature = (configuration, size = 1024, opt_ref_data = None))]
                /// Create a new PyEnsblBasicConf object from a configuration array.
                pub fn new(
                    configuration: PyReadonlyArray2<Float>,
                    size: usize,
                    opt_ref_data: Option<PyReadonlyArray2<Float>>,
                ) -> PyResult<Self>
                {
                    let configuration = array_to_matrix::<Dim<[usize; 2]>, DimSum<Const<$ndim>, U1>, Dyn>(configuration, "configuration")?;
                    let ref_data = match opt_ref_data {
                        Some(ref_data) => {
                            // This array only contains values and not vectors.
                            let matrix = array_to_matrix::<Dim<[usize; 2]>, Const<$nval>, Dyn>(ref_data, "ref_data")?;

                           Some(DVector::from_iterator(matrix.ncols(), matrix.column_iter().map(|col| ObsVec::from(col.clone_owned()))))
                        },
                        None => None
                    };

                    let col_initial = configuration.column(0);
                    let initial = BasicConf::from((col_initial[0].clone(), &col_initial.fixed_rows::<$ndim>(1)));

                    if configuration.ncols() < 2 {
                        return Err(PyValueError::new_err("Invalid configuration array, column length must be larger than 2"));
                    }

                    let configuration = ConfSeries::new(&configuration.column_iter().skip(1).map(|col| {
                        BasicConf::from((col[0].clone(), &col.fixed_rows::<$ndim>(1)))
                    }).collect::<Vec<BasicConf<Float, $ndim>>>());

                    let ensblobs = EnsembleObservations::new(
                        initial,
                        configuration,
                        size,
                        ref_data,
                    );

                    match ensblobs {
                        Some(ensblobs) => Ok(Self(ensblobs)),
                        None => Err(PyValueError::new_err(format!("configuration and optional reference data is mismatched in length"))),
                    }
                }

                /// Return the size of the ensemble.
                pub fn size(&self) -> usize {
                    self.0.size()
                }
            }

            impl From<EnsembleObservations<BasicConf<Float, $ndim>, ObsVec<Float, $nval>>> for [<PyEnsblBasicConf $ndim ObsVec $nval>] {
                fn from(obs_ensbl: EnsembleObservations<BasicConf<Float, $ndim>, ObsVec<Float, $nval>>) -> Self {
                    Self(obs_ensbl)
                }
            }
        }
    };
}

impl_py_ensbl_basicconf_obsvec!(1, 1);
impl_py_ensbl_basicconf_obsvec!(2, 1);
impl_py_ensbl_basicconf_obsvec!(2, 2);
impl_py_ensbl_basicconf_obsvec!(3, 1);
impl_py_ensbl_basicconf_obsvec!(3, 2);
impl_py_ensbl_basicconf_obsvec!(3, 3);
impl_py_ensbl_basicconf_obsvec!(4, 1);
impl_py_ensbl_basicconf_obsvec!(4, 3);

pub use export_py_ensbl_basicconf_obsvec;
