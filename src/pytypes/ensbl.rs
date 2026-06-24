use crate::pytypes::Float;
use numpy::ToPyArray;

/// Export the PyEnsblBasicConfObsVec classes.
#[macro_export]
macro_rules! export_py_ensbl_basicconf_obsvec {
    ($module: expr, $(($ndim: expr, $nval: expr)),+) => {
        $(paste::paste!{
            $module.add_class::<[<PyEnsblBasicConf $ndim ObsVec $nval>]>()?;
        });+
    };
}


/// Implement a PyEnsblBasicConfObsVec class for $ndim dimensions with a $nval dimensional observation vector.
#[macro_export]
macro_rules! impl_py_ensbl_conf_obsvec {
    ($conf_type: ident, $ndim: expr, $nval: expr) => {
        paste::paste! {
            #[derive(Clone)]
            #[doc="PyEnsblVec for " $conf_type "(" $ndim ") and ObsVec("  $nval ")."]
            #[pyo3::pyclass(from_py_object, name = Ensbl $conf_type $ndim ObsVec $nval)]
            pub struct [<PyEnsbl $conf_type $ndim ObsVec $nval>](pub $crate::EnsembleObservations<$crate::conf:: $conf_type<Float, $ndim>, $crate::obs::ObsVec<Float, $nval>>);

            #[pyo3::pymethods]
            impl [<PyEnsbl $conf_type $ndim ObsVec $nval>] {
                /// Return the inner value of ensemble member(s) as a 2D numpy array.
                pub fn get<'py>(&self, py: pyo3::Python<'py>, key: usize) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray2<Float>>> {
                    let matrix = if key < self.0.len() {
                        let column = self.0.output(key);

                        nalgebra::DMatrix::from_iterator(
                            $nval,
                            self.0.len(),
                            column.iter().flat_map(|value| value.iter().cloned()),
                        )
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err("key must be smaller than ensemble size"));
                    };

                    Ok(matrix.transpose().to_pyarray(py))
                }

                /// Return the size of the ensemble.
                pub fn size(&self) -> usize {
                    self.0.size()
                }
            }

            impl From<$crate::EnsembleObservations<$crate::conf:: $conf_type<Float, $ndim>, $crate::obs::ObsVec<Float, $nval>>> for [<PyEnsbl $conf_type $ndim ObsVec $nval>] {
                fn from(obs_ensbl: $crate::EnsembleObservations<$crate::conf:: $conf_type<Float, $ndim>, $crate::obs::ObsVec<Float, $nval>>) -> Self {
                    Self(obs_ensbl)
                }
            }
        }
    };
}

impl_py_ensbl_conf_obsvec!(BasicConf, 1, 1);
impl_py_ensbl_conf_obsvec!(BasicConf, 2, 1);
impl_py_ensbl_conf_obsvec!(BasicConf, 2, 2);
impl_py_ensbl_conf_obsvec!(BasicConf, 3, 1);
impl_py_ensbl_conf_obsvec!(BasicConf, 3, 2);
impl_py_ensbl_conf_obsvec!(BasicConf, 3, 3);

impl_py_ensbl_conf_obsvec!(BasicConfList, 4, 1);

pub use {export_py_ensbl_basicconf_obsvec, impl_py_ensbl_conf_obsvec};

