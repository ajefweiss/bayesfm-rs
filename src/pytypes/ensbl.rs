use crate::{EnsembleObservations, conf::WCSConf, obs::ObsImg, pytypes::Float};
use numpy::ToPyArray;

/// Export the PyEnsblLocationObsVec classes.
#[macro_export]
macro_rules! export_py_ensbl_Location_obsvec {
    ($module: expr, $(($ndim: expr, $nval: expr)),+) => {
        $(paste::paste!{
            $module.add_class::<[<PyEnsblLocation $ndim ObsVec $nval>]>()?;
        });+
    };
}

/// Implement a PyEnsblLocationObsVec class for $ndim dimensions with a $nval dimensional observation vector.
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
                        return Err(pyo3::exceptions::PyValueError::new_err("key must be smaller than the ensemble size"));
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

impl_py_ensbl_conf_obsvec!(Location, 1, 1);
impl_py_ensbl_conf_obsvec!(Location, 2, 1);
impl_py_ensbl_conf_obsvec!(Location, 2, 2);
impl_py_ensbl_conf_obsvec!(Location, 3, 1);
impl_py_ensbl_conf_obsvec!(Location, 3, 2);
impl_py_ensbl_conf_obsvec!(Location, 3, 3);
impl_py_ensbl_conf_obsvec!(Location, 4, 1);

impl_py_ensbl_conf_obsvec!(LocationList, 4, 1);

/// PyEnsblVec for `WCSConf` and `ObsImg`.
#[derive(Clone)]
#[pyo3::pyclass(from_py_object, name = "EnsblWCSConfImg")]
pub struct PyEnsblWCSConfImg(pub EnsembleObservations<WCSConf<Float>, ObsImg<Float>>);

#[pyo3::pymethods]
impl PyEnsblWCSConfImg {
    /// Return the inner value of ensemble member(s) as a 2D numpy array.
    pub fn get<'py>(
        &self,
        py: pyo3::Python<'py>,
        key: usize,
    ) -> pyo3::PyResult<Vec<pyo3::Bound<'py, numpy::PyArray2<Float>>>> {
        let vecmatrix = if key < self.0.len() {
            let column = self.0.output(key);

            Vec::from_iter(column.iter().map(|obs| obs.0.transpose().to_pyarray(py)))
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "key must be smaller than the ensemble size",
            ));
        };

        Ok(vecmatrix)
    }

    /// Return the size of the ensemble.
    pub fn size(&self) -> usize {
        self.0.size()
    }
}

impl From<EnsembleObservations<WCSConf<Float>, ObsImg<Float>>> for PyEnsblWCSConfImg {
    fn from(obs_ensbl: EnsembleObservations<WCSConf<Float>, ObsImg<Float>>) -> Self {
        Self(obs_ensbl)
    }
}

pub use {export_py_ensbl_Location_obsvec, impl_py_ensbl_conf_obsvec};
