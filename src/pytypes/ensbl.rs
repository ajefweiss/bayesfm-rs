use crate::{
    EnsembleObservations,
    conf::{BasicConf, ConfSeries},
    obs::ObsVec,
    pytypes::*,
};
use nalgebra::{Const, DMatrix, DVector, DimSum, Dyn, U1};
use numpy::{PyReadonlyArray2, ToPyArray, ndarray::Dim};
use paste::paste;
use pyo3::{exceptions::PyValueError, prelude::*};

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
                pub fn get<'py>(&self, py: Python<'py>, keys: Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
                    let matrix = if let Ok(key) = keys.extract::<usize>() {
                        let column = self.0.output(key);

                        DMatrix::from_iterator(
                            1,
                            self.0.len(),
                            column.iter().flat_map(|value| value.iter().cloned()),
                        )
                    } else if let Ok(keys) = keys.extract::<Vec<usize>>() {
                        let columns = self.0.outputs(keys.as_slice());

                        DMatrix::<Float>::from_iterator(
                            keys.len(),
                            self.0.len(),
                            columns.iter().flat_map(|view| view.iter().flat_map(|value| value.iter().cloned())),
                        )
                    } else {
                        return Err(PyValueError::new_err("keys argument must be an index or a list of indices"));
                    };

                    Ok(matrix.transpose().to_pyarray(py).into_any())
                }

                #[new]
                #[pyo3(signature = (confs_array, opt_ref_data = None))]
                /// Create a new PyEnsblVec object.
                pub fn new(
                    confs_array: PyReadonlyArray2<Float>,
                    opt_ref_data: Option<PyReadonlyArray2<Float>>,
                ) -> PyResult<Self>
                {
                    let confs_array = array_to_matrix::<Dim<[usize; 2]>, DimSum<DimSum<Const<$ndim>, U1>, Const<$ndim>>, Dyn, U1, DimSum<DimSum<Const<$ndim>, U1>, Const<$ndim>>>(confs_array)?;
                    let ref_data = match opt_ref_data {
                        Some(ref_data) => {
                            // This array only contains values and not vectors.
                            let raw_array = array_to_matrix::<Dim<[usize; 2]>, Const<$nval>, Dyn, U1, Const<$nval>>(ref_data)?;

                           Some(DVector::from_iterator(raw_array.ncols(), raw_array.column_iter().map(|col| ObsVec::from(col.clone_owned()))))
                        },
                        None => None
                    };

                    let col_initial = confs_array.column(0);
                    let initial = BasicConf::from((col_initial[0].clone(), col_initial.fixed_rows::<$ndim>(1), col_initial.fixed_rows::<$ndim>(1 + $ndim)));

                    let size = confs_array.ncols() - 1;

                    if size < 1 {
                        return Err(PyValueError::new_err("Invalid configuration array, column length must be larger than 2"));
                    }

                    let configuration = ConfSeries::new(&confs_array.column_iter().skip(1).map(|col| {
                        BasicConf::from((col[0].clone(), col.fixed_rows::<$ndim>(1), col.fixed_rows::<$ndim>(1 + $ndim)))
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
impl_py_ensbl_basicconf_obsvec!(3, 4);

pub use export_py_ensbl_basicconf_obsvec;
