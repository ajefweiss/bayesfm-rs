/// A macro that generates a function that creates a new custom filter.
#[macro_export]
macro_rules! py_add_model_functions {
    ($model: ty, $name: ident, $nparams: expr) => {
        paste::paste! {
            #[pyo3::pymethods]
            impl $name {
                /// Return the names of the model parameters.
                #[classmethod]
                pub fn names(_cls: &pyo3::Bound<pyo3::types::PyType>,) -> Vec<String> {
                    $model::<Float, prodef::MultivariateDensity<Float, nalgebra::Const<$nparams>>>::PARAM_NAMES.iter().map(|name| name.to_string()).collect()
                }
            }
        }
    };
}

pub use py_add_model_functions;
