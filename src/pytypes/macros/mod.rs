mod filter;
mod functions;
mod simulation;

/// Convert a python iterator to an iterator of a specific type.
#[macro_export]
macro_rules! py_any_iterator {
    ($iterator: expr, $type: ty) => {
        match $iterator.try_iter() {
            Ok(value) => value.map(|py_obj| py_obj.unwrap().extract::<$type>().unwrap()),
            Err(..) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Values argument must be iterable",
                ))
            }
        }
    };
}

/// Select an error metric function based on the provided metric name.
#[macro_export]
macro_rules! py_select_error_metric {
    ($metric: expr, $type: ty) => {
        match $metric.to_lowercase().as_str() {
            "valid" => |o1: &[$type], o2: &[$type]| {
                bayesfm::obs::ov_error(o1, o2, bayesfm::obs::ObsVecMetric::Valid)
            },
            "nchisq" => |o1: &[$type], o2: &[$type]| {
                bayesfm::obs::ov_error(o1, o2, bayesfm::obs::ObsVecMetric::NChiSq)
            },
            "rmse" => |o1: &[$type], o2: &[$type]| {
                bayesfm::obs::ov_error(o1, o2, bayesfm::obs::ObsVecMetric::RMSE)
            },
            "rmspe" => |o1: &[$type], o2: &[$type]| {
                bayesfm::obs::ov_error(o1, o2, bayesfm::obs::ObsVecMetric::RMSPE)
            },
            "nrmse" => |o1: &[$type], o2: &[$type]| {
                bayesfm::obs::ov_error(o1, o2, bayesfm::obs::ObsVecMetric::NRMSE)
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported metric",
                ))
            }
        }
    };
}

/// Select an error metric function based on the provided metric name.
#[macro_export]
macro_rules! py_select_error_special_metric {
    ($metric: expr, $type: ty) => {
        match $metric.to_lowercase().as_str() {
            "dtw" => |o1: &[$type], o2: &[$type]| {
                ov_error(o1, o2, bayesfm::obs::ObsVecSpecialMetric::DTW)
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported metric",
                ))
            }
        }
    };
}

/// Unroll model errors into Python exceptions.
#[macro_export]
macro_rules! py_unroll_model_errors {
    ($match: expr) => {
        match $match {
            Ok(result) => Ok(result),
            Err(err) => match err {
                bayesfm::ModelError::Sampling => Err(pyo3::exceptions::PyRuntimeError::new_err(
                    ("Failed to sample the parameter space"),
                )),
                bayesfm::ModelError::Coordinates(_) => {
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Failed to perform coordinate transformation",
                    ))
                }
                _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Unhandled model error",
                )),
            },
        }
    };
}

/// Unroll filter errors into Python exceptions.
#[macro_export]
macro_rules! py_unroll_filter_errors {
    ($match: expr) => {
        match $match {
            Ok(result) => Ok(result),
            Err(err) => match err {
                bayesfm::methods::filters::FilterError::TimeLimit { limit, .. } => {
                    Err(pyo3::exceptions::PyRuntimeError::new_err((
                        "The time limit of",
                        limit,
                        "seconds was exceeded",
                    )))
                }
                // bayesfm::methods::filters::FilterError::TimeLimitPredicted {
                //     predicted,
                //     limit,
                //     ..
                // } => Err(pyo3::exceptions::PyRuntimeError::new_err((
                //     "The time limit of",
                //     limit,
                //     "seconds was predicted to be exceeded with ",
                //     predicted,
                //     "seconds",
                // ))),
                // bayesfm::methods::filters::FilterError::Model(model_err) => {
                //     bayesfm::py_unroll_model_errors!(Err(model_err))
                // }
                // bayesfm::methods::filters::FilterError::EffectiveParticles(count) => {
                //     Err(pyo3::exceptions::PyRuntimeError::new_err((
                //         "Insufficient effective particles (",
                //         count,
                //         ")",
                //     )))
                // }
                _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Unhandled particle filter error",
                )),
            },
        }
    };
}

/// Unwrap a PyResult, returning an error with a custom message on [`Err`].
#[macro_export]
macro_rules! py_unwrap {
    ($expr: expr, $text: literal) => {
        match $expr {
            Some(value) => value,
            None => return Err(pyo3::exceptions::PyValueError::new_err($text)),
        }
    };
}

pub use {
    filter::py_add_model_filter, functions::py_add_model_functions, py_any_iterator,
    py_select_error_metric, py_select_error_special_metric, py_unroll_filter_errors,
    py_unroll_model_errors, py_unwrap, simulation::py_add_model_simulation,
};
