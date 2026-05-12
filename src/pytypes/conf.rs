use crate::{
    conf::{BasicConf, ConfSeries, ConfTime},
    pytypes::{Float, array_to_matrix},
};
use nalgebra::{Const, Dyn, OMatrix, U1};
use numpy::{PyReadonlyArray2, ndarray::Dim};
use paste::paste;
use pyo3::{prelude::*, types::PyType};

/// Export the PyBasicConfSeries classes.
#[macro_export]
macro_rules! export_py_basic_conf_series {
    ($module: expr, $($ndim: expr),+) => {
        $(paste::paste!{
            $module.add_class::<[<PyBasicConf $ndim>]>()?;
        });+

        $(paste::paste!{
            $module.add_class::<[<PyBasicConf $ndim Series>]>()?;
        });+
    };
}

macro_rules! impl_py_basic_conf_series {
    ($ndim: expr) => {
        paste! {
            #[pyclass(from_py_object, name = BasicConf $ndim)]
            #[derive(Clone)]
            #[doc="PyBasicConf for n="  $ndim]
            pub struct [<PyBasicConf $ndim>](pub BasicConf<Float, $ndim>);

            #[pyclass(from_py_object, name = BasicConf $ndim Series)]
            #[derive(Clone)]
            #[doc="PyBasicConfSeries for n="  $ndim]
            pub struct [<PyBasicConf $ndim Series>](pub ConfSeries<BasicConf<Float, $ndim>>);

            #[pymethods]
            impl [<PyBasicConf $ndim>] {
                /// Create a new configuration.
                #[new]
                #[pyo3(signature = (timestamp, opt_position = None))]
                pub fn new(
                    timestamp: Float,
                    opt_position: Option<PyReadonlyArray2<Float>>,
                ) -> PyResult<Self> {
                    let position = match opt_position {
                        Some(position) => array_to_matrix::<Dim<[usize; 2]>, Const<$ndim>, U1>(position, "position")?,
                        None => OMatrix::zeros_generic(Const::<$ndim>, Const::<1>),
                    };

                    Ok([<PyBasicConf $ndim>](
                       BasicConf::new(timestamp, position)
                    ))
                }
            }

            #[pymethods]
            impl [<PyBasicConf $ndim Series>] {
                /// Combine two configurations.
                #[classmethod]
                pub fn combine(_cls: &Bound<PyType>, conf_a: &[<PyBasicConf $ndim Series>], conf_b: &[<PyBasicConf $ndim Series>]) -> PyResult<Self> {
                    let conf = conf_a.0.clone() + conf_b.0.clone();

                    Ok([<PyBasicConf $ndim Series>](conf))
                }

                /// Return the number of observations.
                pub fn count(&self) -> usize {
                    self.0.count()
                }

                /// Create a new configuration.
                #[new]
                #[pyo3(signature = (timestamps, opt_position = None))]
                pub fn new(
                    timestamps: Vec<Float>,
                    opt_position: Option<PyReadonlyArray2<Float>>,
                ) -> PyResult<Self> {
                    let count = timestamps.len();

                    let position = match opt_position {
                        Some(position) => array_to_matrix::<Dim<[usize; 2]>, Const<$ndim>, Dyn>(position, "position")?,
                        None => OMatrix::zeros_generic(Const::<$ndim>, Dyn(count)),
                    };

                    Ok([<PyBasicConf $ndim Series>](
                        timestamps
                            .iter()
                            .zip(position.column_iter())
                            .map(|(ts, pos)| BasicConf::new(*ts, pos.clone_owned()))
                            .collect(),
                    ))
                }

                // pub fn positions<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Float>>> {
                //     let positions = self.0.positions();
                //     positions.map(|pos| pos.to_pyarray(py))
                // }


                //     /// Return a subset of the observations given by the provided indices.
                //     pub fn subset(&self, indices: Vec<usize>) -> PyResult<Self> {
                //         match self {
                //             PyObs::Obs0(obs) => Ok(PyObs::Obs0(PyObs0(obs.0.subset(&indices)))),
                //             PyObs::Obs1(obs) => Ok(PyObs::Obs1(PyObs1(obs.0.subset(&indices)))),
                //             PyObs::Obs2(obs) => Ok(PyObs::Obs2(PyObs2(obs.0.subset(&indices)))),
                //             PyObs::Obs3(obs) => Ok(PyObs::Obs3(PyObs3(obs.0.subset(&indices)))),
                //             PyObs::ObsCam(obs) => Ok(PyObs::ObsCam(PyObsCam(obs.0.subset(&indices)))),
                //         }
                //     }

                /// Return the observation timestamps as a vector.
                pub fn timestamps(&self) -> Vec<Float> {
                    self.0.clone().into_iter().map(|conf| conf.timestamp()).collect()
                }

                /// Return the uncombined indices.
                pub fn uncombined_indices(&self) -> Vec<usize> {
                    self.0.uncombined_indices()
                }
            }
        }
    };
}

impl_py_basic_conf_series!(1);
impl_py_basic_conf_series!(2);
impl_py_basic_conf_series!(3);
impl_py_basic_conf_series!(4);

pub use export_py_basic_conf_series;
