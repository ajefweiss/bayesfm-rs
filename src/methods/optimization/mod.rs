//! Methods and methods for solving nonlinear equations and optimization problems.

mod newton;

use nalgebra::{DefaultAllocator, Dim, OVector, RealField, allocator::Allocator};
pub use newton::*;

/// Result for an optimization containing the optimized vector and convergence diagnostics.
#[derive(Clone, Debug)]
pub struct OptimizationResult<T: RealField, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// The optimized vector.
    pub solution: OVector<T, D>,
    /// The final residual norm squared
    pub residual_norm_sq: T,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
}
