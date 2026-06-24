use crate::{
    conf::{BasicConf, BasicConfList, ConfSeries, ConfTime},
    pytypes::{Float, array_to_matrix},
};
use nalgebra::{Const, Dyn, OMatrix, U1};
use numpy::{PyReadonlyArray2, ndarray::Dim};
use paste::paste;
use pyo3::{prelude::*, types::PyType};


macro_rules! impl_py_basic_conf {
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


                /// Return a subset of the observations given by the provided indices.
                pub fn subset(&self, indices: Vec<usize>) -> PyResult<Self> {
                    match self.0.subset(&indices) {
                        Some(subset) => Ok(Self(subset)),
                        _ => Err(pyo3::exceptions::PyValueError::new_err("invalid indices")),
                    }
                }

                /// Sort the underlying configurations.
                pub fn sort(&mut self) {
                    self.0.sort();
                }

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

macro_rules! impl_py_basic_list_conf {
    ($ndim: expr) => {
        paste! {
            #[pyclass(from_py_object, name = BasicConfList $ndim)]
            #[derive(Clone)]
            #[doc="PyBasicConfList for n="  $ndim]
            pub struct [<PyBasicConfList $ndim>](pub BasicConfList<Float, $ndim>);

            #[pyclass(from_py_object, name = BasicConfList $ndim Series)]
            #[derive(Clone)]
            #[doc="PyBasicConfListSeries for n="  $ndim]
            pub struct [<PyBasicConfList $ndim Series>](pub ConfSeries<BasicConfList<Float, $ndim>>);

            #[pymethods]
            impl [<PyBasicConfList $ndim>] {
                /// Create a new configuration.
                #[new]
                #[pyo3(signature = (timestamp, opt_position = None))]
                pub fn new(
                    timestamp: Float,
                    opt_position: Option<PyReadonlyArray2<Float>>,
                ) -> PyResult<Self> {
                    let position = match opt_position {
                        Some(position) => array_to_matrix::<Dim<[usize; 2]>, Const<$ndim>, Dyn>(position, "position")?,
                        None => OMatrix::zeros_generic(Const::<$ndim>, Dyn(0)),
                    };

                    Ok([<PyBasicConfList $ndim>](
                       BasicConfList::new(timestamp, position)
                    ))
                }
            }

            #[pymethods]
            impl [<PyBasicConfList $ndim Series>] {
                /// Combine two configurations.
                #[classmethod]
                pub fn combine(_cls: &Bound<PyType>, conf_a: &[<PyBasicConfList $ndim Series>], conf_b: &[<PyBasicConfList $ndim Series>]) -> PyResult<Self> {
                    let conf = conf_a.0.clone() + conf_b.0.clone();

                    Ok([<PyBasicConfList $ndim Series>](conf))
                }

                /// Return the number of observations.
                pub fn count(&self) -> usize {
                    self.0.count()
                }

                /// Create a new configuration.
                #[new]
                #[pyo3(signature = (timestamps, opt_positions = None))]
                pub fn new(
                    timestamps: Vec<Float>,
                    opt_positions: Option<Vec<PyReadonlyArray2<Float>>>,
                ) -> PyResult<Self> {
                    let positions = match opt_positions {
                        Some(positions) => positions.into_iter().map(|pos| array_to_matrix::<Dim<[usize; 2]>, Const<$ndim>, Dyn>(pos, "position")).collect::<PyResult<Vec<_>>>()?,
                        None => timestamps.iter().map(|_| OMatrix::zeros_generic(Const::<$ndim>, Dyn(1))).collect(),
                    };

                    Ok([<PyBasicConfList $ndim Series>](ConfSeries::from_iter(
                        timestamps
                            .iter()
                            .zip(positions)
                            .map(|(ts, pos)| BasicConfList::new(*ts, pos))
                    )))
                }

                /// Sort the underlying configurations.
                pub fn sort(&mut self) {
                    self.0.sort();
                }

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

impl_py_basic_conf!(1);
impl_py_basic_conf!(2);
impl_py_basic_conf!(3);

impl_py_basic_list_conf!(4);
