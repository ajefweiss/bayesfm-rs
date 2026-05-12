//! Python types to be used with `pyo3`.
//!
//! The types in this module are designed to be used with the `pyo3` library, which allows for seamless interoperability between Rust and Python.

mod conf;
mod ensbl;
mod macros;
mod noise;

pub use conf::*;
pub use ensbl::*;
pub use macros::*;
pub use noise::*;

#[cfg(feature = "pyo3_f32")]
type Float = f32;
#[cfg(not(feature = "pyo3_f32"))]
type Float = f64;

use nalgebra::{DefaultAllocator, Dim, Dyn, OMatrix, allocator::Allocator};
use numpy::{PyReadonlyArray, ndarray::Dimension};
use pyo3::PyResult;

/// Convert a PyArray to nalgebra [`OMatrix`].
///
/// The strides are always assumed to be dynamic.
pub fn array_to_matrix<D, R, C>(
    array: PyReadonlyArray<Float, D>,
    name: &str,
) -> PyResult<OMatrix<Float, R, C>>
where
    D: Dimension,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
{
    match array.try_as_matrix::<R, C, Dyn, Dyn>() {
        Some(value) => Ok(value.clone_owned()),
        None => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Conversion of a numpy array \"{}\" to nalgebra matrix failed",
            name
        ))),
    }
}
