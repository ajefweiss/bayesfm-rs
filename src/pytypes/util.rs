use crate::pytypes::Float;
use nalgebra::{DefaultAllocator, Dim, OMatrix, allocator::Allocator};
use numpy::{PyReadonlyArray, ndarray::Dimension};
use pyo3::PyResult;

/// Convert a PyArray to nalgebra [`OMatrix`].
pub fn array_to_matrix<D, R, C, RStride, CStride>(
    array: PyReadonlyArray<Float, D>,
) -> PyResult<OMatrix<Float, R, C>>
where
    D: Dimension,
    R: Dim,
    C: Dim,
    RStride: Dim,
    CStride: Dim,
    DefaultAllocator: Allocator<R, C>,
{
    match array.try_as_matrix::<R, C, RStride, CStride>() {
        Some(value) => Ok(value.clone_owned()),
        None => Err(pyo3::exceptions::PyValueError::new_err(
            "Conversion of a numpy array to nalgebra matrix failed",
        )),
    }
}

/// Convert a python iterator to an iterator of a specific type.
#[macro_export]
macro_rules! py_any_iterator {
    ($iterator: expr, $type: ty) => {
        match $iterator.try_iter() {
            Ok(value) => value.map(|py_obj| match py_obj {
                Ok(py_obj_inner) => match py_obj_inner.extract::<$type>() {
                    Ok(val) => val,
                    Err(_) => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Iterator values must be convertible to the target type \"{}\"",
                            stringify!($type)
                        )))
                    }
                },
                Err(_) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Values argument must be iterable",
                    ))
                }
            }),
            Err(..) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Values argument must be iterable",
                ))
            }
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

pub use {py_any_iterator, py_unwrap};
