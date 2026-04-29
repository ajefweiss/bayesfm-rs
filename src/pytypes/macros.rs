/// Select an error metric function based on the provided metric name.
#[macro_export]
macro_rules! select_error_metric {
    ($metric: expr, $type: ty) => {
        match $metric.to_lowercase().as_str() {
            "valid" => |o1: &[$type], o2: &[$type]| ov_error(o1, o2, ObsVecMetric::Valid),
            "nchisq" => |o1: &[$type], o2: &[$type]| ov_error(o1, o2, ObsVecMetric::NChiSq),
            "rmse" => |o1: &[$type], o2: &[$type]| ov_error(o1, o2, ObsVecMetric::RMSE),
            "rmspe" => |o1: &[$type], o2: &[$type]| ov_error(o1, o2, ObsVecMetric::RMSPE),
            "nrmse" => |o1: &[$type], o2: &[$type]| ov_error(o1, o2, ObsVecMetric::NRMSE),
            "dtw" => |o1: &[$type], o2: &[$type]| ov_error(o1, o2, ObsVecMetric::DTW),
            _ => return Err(PyValueError::new_err("Unsupported metric")),
        }
    };
}

/// Unroll model errors into Python exceptions.
#[macro_export]
macro_rules! unroll_model_errors {
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
macro_rules! unroll_filter_errors {
    ($match: expr) => {
        match $match {
            Ok(result) => Ok(result),
            Err(err) => match err {
                bayesfm::methods::filters::FilterError::TimeLimit { limit, .. } => Err(
                    PyRuntimeError::new_err(("The time limit of", limit, "seconds was exceeded")),
                ),
                bayesfm::methods::filters::FilterError::TimeLimitPredicted {
                    predicted,
                    limit,
                    ..
                } => Err(PyRuntimeError::new_err((
                    "The time limit of",
                    limit,
                    "seconds was predicted to be exceeded with ",
                    predicted,
                    "seconds",
                ))),
                bayesfm::methods::filters::FilterError::Model(model_err) => {
                    unroll_model_errors!(Err(model_err))
                }
                bayesfm::methods::filters::FilterError::EffectiveParticles(count) => Err(
                    PyRuntimeError::new_err(("Insufficient effective particles (", count, ")")),
                ),
                _ => Err(PyRuntimeError::new_err("Unhandled particle filter error")),
            },
        }
    };
}

pub use {select_error_metric, unroll_filter_errors, unroll_model_errors};
