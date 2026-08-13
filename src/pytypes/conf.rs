use crate::{
    conf::{ConfSeries, ConfTime, Location, LocationList, WCSConf},
    pytypes::{Float, array_to_matrix},
};
use nalgebra::{Const, Dyn, OMatrix, U1};
use numpy::{PyReadonlyArray2, ndarray::Dim};
use paste::paste;
use pyo3::{prelude::*, types::PyType};
use wcs::WCSParams;

macro_rules! impl_py_loc_conf {
    ($ndim: expr) => {
        paste! {
            #[pyclass(from_py_object, name = Location $ndim)]
            #[derive(Clone)]
            #[doc="PyLocation for n="  $ndim]
            pub struct [<PyLocation $ndim>](pub Location<Float, $ndim>);

            #[pyclass(from_py_object, name = Location $ndim Series)]
            #[derive(Clone)]
            #[doc="PyLocationSeries for n="  $ndim]
            pub struct [<PyLocation $ndim Series>](pub ConfSeries<Location<Float, $ndim>>);

            #[pymethods]
            impl [<PyLocation $ndim>] {
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

                    Ok([<PyLocation $ndim>](
                       Location::new(timestamp, position)
                    ))
                }
            }

            #[pymethods]
            impl [<PyLocation $ndim Series>] {
                /// Combine two configurations.
                #[classmethod]
                pub fn combine(_cls: &Bound<PyType>, conf_a: &[<PyLocation $ndim Series>], conf_b: &[<PyLocation $ndim Series>]) -> PyResult<Self> {
                    let conf = conf_a.0.clone() + conf_b.0.clone();

                    Ok([<PyLocation $ndim Series>](conf))
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

                    Ok([<PyLocation $ndim Series>](
                        timestamps
                            .iter()
                            .zip(position.column_iter())
                            .map(|(ts, pos)| Location::new(*ts, pos.clone_owned()))
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

macro_rules! impl_py_loc_list_conf {
    ($ndim: expr) => {
        paste! {
            #[pyclass(from_py_object, name = LocationList $ndim)]
            #[derive(Clone)]
            #[doc="PyLocationList for n="  $ndim]
            pub struct [<PyLocationList $ndim>](pub LocationList<Float, $ndim>);

            #[pyclass(from_py_object, name = LocationList $ndim Series)]
            #[derive(Clone)]
            #[doc="PyLocationListSeries for n="  $ndim]
            pub struct [<PyLocationList $ndim Series>](pub ConfSeries<LocationList<Float, $ndim>>);

            #[pymethods]
            impl [<PyLocationList $ndim>] {
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

                    Ok([<PyLocationList $ndim>](
                       LocationList::new(timestamp, position)
                    ))
                }
            }

            #[pymethods]
            impl [<PyLocationList $ndim Series>] {
                /// Combine two configurations.
                #[classmethod]
                pub fn combine(_cls: &Bound<PyType>, conf_a: &[<PyLocationList $ndim Series>], conf_b: &[<PyLocationList $ndim Series>]) -> PyResult<Self> {
                    let conf = conf_a.0.clone() + conf_b.0.clone();

                    Ok([<PyLocationList $ndim Series>](conf))
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

                    Ok([<PyLocationList $ndim Series>](ConfSeries::from_iter(
                        timestamps
                            .iter()
                            .zip(positions)
                            .map(|(ts, pos)| LocationList::new(*ts, pos))
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

impl_py_loc_conf!(1);
impl_py_loc_conf!(2);
impl_py_loc_conf!(3);
impl_py_loc_conf!(4);

impl_py_loc_list_conf!(4);

#[pyclass(from_py_object, name = "WCSConf")]
#[derive(Clone)]
#[doc = "PyWCSConf"]
pub struct PyWCSConf(pub WCSConf<Float>);

#[pyclass(from_py_object, name = "WCSConfSeries")]
#[derive(Clone)]
#[doc = "PyWCSConfSeries"]
pub struct PyWCSConfSeries(pub ConfSeries<WCSConf<Float>>);

#[pymethods]
impl PyWCSConf {
    /// Create a new configuration.
    #[new]
    pub fn new(
        timestamp: Float,
        position: PyReadonlyArray2<Float>,
        params: String,
        flags: Option<Vec<Float>>,
    ) -> PyResult<Self> {
        let position = array_to_matrix::<Dim<[usize; 2]>, Const<3>, U1>(position, "position")?;
        let params: WCSParams = match serde_json5::from_str(&params) {
            Ok(value) => value,
            Err(err) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to parse WCSParams from JSON: {}",
                    err
                )));
            }
        };

        Ok(PyWCSConf(WCSConf::new(timestamp, position, params, flags)))
    }
}

#[pymethods]
impl PyWCSConfSeries {
    /// Combine two configurations.
    #[classmethod]
    pub fn combine(
        _cls: &Bound<PyType>,
        conf_a: &PyWCSConfSeries,
        conf_b: &PyWCSConfSeries,
    ) -> PyResult<Self> {
        let conf = conf_a.0.clone() + conf_b.0.clone();

        Ok(PyWCSConfSeries(conf))
    }

    /// Return the number of observations.
    pub fn count(&self) -> usize {
        self.0.count()
    }

    /// Create a new configuration.
    #[new]
    pub fn new(
        timestamps: Vec<Float>,
        position: PyReadonlyArray2<Float>,
        params: Vec<String>,
        flags: Option<Vec<Float>>,
    ) -> PyResult<Self> {
        let position = array_to_matrix::<Dim<[usize; 2]>, Const<3>, Dyn>(position, "position")?;
        let mut wcsparams: Vec<Option<WCSParams>> = vec![None; params.len()]; // Placeholder for WCSParams, will be filled in below

        wcsparams
            .iter_mut()
            .zip(params.iter())
            .for_each(|(wcsp, string)| *wcsp = serde_json5::from_str(string).ok());

        if wcsparams.iter().any(|wcsp| wcsp.is_none()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Failed to parse one or more WCSParams from JSON",
            ));
        }

        Ok(PyWCSConfSeries(
            timestamps
                .iter()
                .zip(position.column_iter())
                .zip(wcsparams.iter())
                .map(|((ts, pos), wcs)| {
                    WCSConf::new(
                        *ts,
                        pos.clone_owned(),
                        wcs.as_ref().unwrap().clone(),
                        flags.clone(),
                    )
                })
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
        self.0
            .clone()
            .into_iter()
            .map(|conf| conf.timestamp())
            .collect()
    }

    /// Return the uncombined indices.
    pub fn uncombined_indices(&self) -> Vec<usize> {
        self.0.uncombined_indices()
    }
}
