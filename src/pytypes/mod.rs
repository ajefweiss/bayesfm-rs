//! Python types to be used with `pyo3`.
//!
//! The types in this module are designed to be used with the `pyo3` library, which allows for seamless interoperability between Rust and Python.

mod conf;
mod ensbl;
mod macros;
mod noise;
mod util;

pub use conf::*;
pub use ensbl::*;
pub use macros::*;
pub use noise::*;
pub use util::*;

#[cfg(feature = "pyo3_f32")]
type Float = f32;
#[cfg(not(feature = "pyo3_f32"))]
type Float = f64;
