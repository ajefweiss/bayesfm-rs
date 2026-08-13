use crate::methods::optimization::OptimizationResult;
use nalgebra::{DefaultAllocator, Dim, DimName, Dyn, OVector, RealField, allocator::Allocator};
use std::fmt::Debug;

/// Configuration parameters for the damped Newton optimization method.
///
/// This struct encapsulates all tunable parameters for the generalized damped Newton method,
/// allowing for flexible configuration across different optimization scenarios.
#[derive(Clone, Debug)]
pub struct DampedNewtonConfig<T: RealField> {
    /// Convergence tolerance for the residual norm. Optimization stops when residual < tolerance^2
    pub tolerance: T,
    /// Finite difference step size for computing numerical derivatives
    pub delta_h: T,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Maximum number of backtracking steps in line search
    pub max_backtrack: usize,
    /// Factor to reduce step size by during backtracking (typically 0.5)
    pub backtrack_factor: T,
}

impl<T: RealField> Default for DampedNewtonConfig<T> {
    /// Create a new configuration with default values suitable for most optimization problems.
    ///
    /// Defaults:
    /// - tolerance: 1e-10
    /// - delta_h: 1e-8
    /// - max_iterations: 10
    /// - max_backtrack: 10
    /// - backtrack_factor: 0.5
    fn default() -> Self {
        Self {
            tolerance: T::from_f64(1e-10).unwrap(),
            delta_h: T::from_f64(1e-8).unwrap(),
            max_iterations: 10,
            max_backtrack: 10,
            backtrack_factor: T::from_f64(0.5).unwrap(),
        }
    }
}

/// Generalized damped Newton method for vector optimization problems.
///
/// Solves the optimization problem: minimize f(x) = ||residual(x)||^2
/// where residual is a vector-valued function that measures how far the current
/// solution is from the target.
///
/// The method performs:
/// 1. Numerical differentiation to compute the Jacobian of residual(x)
/// 2. Computes the gradient of f(x) = ||residual(x)||^2 as: ∇f = 2 * J^T * residual
/// 3. Computes Newton step: dx = -H^{-1} * ∇f where H is the approximate Hessian
/// 4. Backtracking line search with step size halving for robustness
///
/// # Arguments
/// - `residual_fn`: A closure that computes the residual vector for a given state.
/// - `initial_guess`: Starting point for the optimization
/// - `config`: Configuration parameters for the optimizer
///
/// # Returns
/// An `OptimizationResult` containing:
/// - `solution`: The optimized variable vector
/// - `residual_norm_sq`: The final residual norm squared
/// - `iterations`: Number of iterations performed
/// - `converged`: Whether convergence criterion was met
///
/// # Example
/// ```ignore
/// use nalgebra::Vector2;
/// let config = DampedNewtonConfig::default();
/// let initial_guess = Vector2::new(1.0, 1.0);
/// let result = damped_newton(
///     |x: &Vector2<f64>| {
///         Some(x - &Vector2::new(2.0, 3.0))  // Solve x = [2, 3]
///     },
///     initial_guess,
///     &config
/// );
/// ```
pub fn damped_newton<T, D, F>(
    residual_fn: F,
    initial_guess: OVector<T, D>,
    config: &DampedNewtonConfig<T>,
) -> Option<OptimizationResult<T, D>>
where
    T: RealField + Debug,
    D: Dim + DimName,
    DefaultAllocator:
        Allocator<D> + Allocator<D, D> + Allocator<Dyn> + Allocator<Dyn, D> + Allocator<Dyn, Dyn>,
    F: Fn(&OVector<T, D>) -> Option<OVector<T, D>>,
{
    let mut x_current = initial_guess;
    let mut iterations = 0;
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();
    let epsilon_gradient = T::from_f64(1e-15).unwrap();

    for _ in 0..config.max_iterations {
        iterations += 1;

        // Evaluate residual at current point
        let residual = residual_fn(&x_current)?;
        let residual_norm_sq = residual.norm_squared();

        // Check convergence on residual
        if residual_norm_sq < config.tolerance.clone() * config.tolerance.clone() {
            return Some(OptimizationResult {
                solution: x_current,
                residual_norm_sq,
                iterations,
                converged: true,
            });
        }

        // Compute Jacobian via numerical differentiation
        let dim = x_current.len();
        let mut jacobian = nalgebra::DMatrix::<T>::zeros(residual.len(), dim);

        for j in 0..dim {
            let mut x_pert = x_current.clone();
            x_pert[j] += config.delta_h.clone();
            let residual_pert = residual_fn(&x_pert)?;
            let col = (&residual_pert - &residual) / config.delta_h.clone();
            jacobian.set_column(j, &col);
        }

        // Compute gradient: ∇f = 2 * J^T * residual
        let residual_dyn = nalgebra::DVector::from_column_slice(residual.as_slice());
        let gradient = &jacobian.transpose() * &residual_dyn * two.clone();

        // Guard against a flat or degenerate gradient
        let grad_norm = gradient.norm();
        if grad_norm < epsilon_gradient {
            return Some(OptimizationResult {
                solution: x_current,
                residual_norm_sq,
                iterations,
                converged: false,
            });
        }

        // Compute approximate Hessian: H ≈ 2 * J^T * J
        let hessian = &jacobian.transpose() * &jacobian * two.clone();

        // Compute Newton step: dx = -H^{-1} * ∇f
        let dx_dyn = match hessian.lu().solve(&(-gradient.clone())) {
            Some(step) => step,
            None => {
                // Singular Hessian, return current point
                return Some(OptimizationResult {
                    solution: x_current,
                    residual_norm_sq,
                    iterations,
                    converged: false,
                });
            }
        };

        // Convert back to fixed-size vector
        let dx = OVector::<T, D>::from_column_slice(dx_dyn.as_slice());

        // Backtracking line search: halve the step until the residual norm squared decreases
        let mut alpha = one.clone();
        for _ in 0..config.max_backtrack {
            let x_trial = x_current.clone() + dx.clone() * alpha.clone();
            let residual_trial = match residual_fn(&x_trial) {
                Some(r) => r,
                None => {
                    // Evaluation failed, reduce step size
                    alpha = alpha.clone() * config.backtrack_factor.clone();
                    continue;
                }
            };
            let residual_trial_norm_sq = residual_trial.norm_squared();

            if residual_trial_norm_sq < residual_norm_sq {
                break;
            }

            alpha = alpha.clone() * config.backtrack_factor.clone();
        }

        let dx_damped = dx.clone() * alpha.clone();

        // Convergence check on step size
        let step_norm = dx_damped.norm();
        if step_norm < config.tolerance.clone() {
            x_current += dx_damped;
            let final_residual = residual_fn(&x_current)?;
            return Some(OptimizationResult {
                solution: x_current,
                residual_norm_sq: final_residual.norm_squared(),
                iterations,
                converged: true,
            });
        }

        x_current += dx_damped;
    }

    // Max iterations reached
    let final_residual = residual_fn(&x_current)?;
    Some(OptimizationResult {
        solution: x_current,
        residual_norm_sq: final_residual.norm_squared(),
        iterations,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector1, Vector3};

    #[test]
    fn test_damped_newton_1d() {
        // Test solving x^2 - 4 = 0, which has solution x = 2
        let config = DampedNewtonConfig::default();
        let initial_guess = Vector1::new(1.0);

        let result = damped_newton(
            |x: &Vector1<f64>| {
                // Residual: x^2 - 4
                Some(Vector1::new(x[0] * x[0] - 4.0))
            },
            initial_guess,
            &config,
        );

        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.converged);
        assert!((result.solution[0] - 2.0).abs() < 1e-6);
        assert!(result.residual_norm_sq < 1e-10);
    }

    #[test]
    fn test_damped_newton_3d() {
        // Test solving the system:
        // x + y + z = 6
        // x^2 + y^2 + z^2 = 14
        // x*y*z = 6
        // Which has solution approximately (1, 2, 3)
        let config = DampedNewtonConfig::default();
        let initial_guess = Vector3::new(1.5, 2.5, 2.0);

        let result = damped_newton(
            |x: &Vector3<f64>| {
                let eq1 = x[0] + x[1] + x[2] - 6.0;
                let eq2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 14.0;
                let eq3 = x[0] * x[1] * x[2] - 6.0;
                Some(Vector3::new(eq1, eq2, eq3))
            },
            initial_guess,
            &config,
        );

        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.converged);

        // Check residual is small
        assert!(result.residual_norm_sq < 1e-10);

        // Verify the solution satisfies the equations
        let sol = &result.solution;
        assert!((sol[0] + sol[1] + sol[2] - 6.0).abs() < 1e-6);
        assert!((sol[0] * sol[0] + sol[1] * sol[1] + sol[2] * sol[2] - 14.0).abs() < 1e-6);
        assert!((sol[0] * sol[1] * sol[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_damped_newton_config_default() {
        // Test that default config has expected values
        let config = DampedNewtonConfig::<f64>::default();
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.max_backtrack, 10);
        assert!((config.tolerance - 1e-10).abs() < 1e-15);
        assert!((config.delta_h - 1e-8).abs() < 1e-15);
        assert!((config.backtrack_factor - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_damped_newton_1d_linear() {
        // Test solving 2*x - 6 = 0, which has solution x = 3
        let config = DampedNewtonConfig::default();
        let initial_guess = Vector1::new(0.0);

        let result = damped_newton(
            |x: &Vector1<f64>| {
                // Residual: 2*x - 6
                Some(Vector1::new(2.0 * x[0] - 6.0))
            },
            initial_guess,
            &config,
        );

        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.converged);
        assert!((result.solution[0] - 3.0).abs() < 1e-6);
    }
}
