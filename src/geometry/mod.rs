//! # Curvilinear coordinate systems and model geometries.
//!
//! This module introduces the [`Geometry`] trait, which is the trait that is shared by all curvilinear coordinate systems that define a model geometry.
//! The trait provides bi-directional coordinate transformation functions, and methods that compute the covariant and contravariant basis vectors.
//! Note that the basis vectors of the coordinate systems are not necessarily orthonormal.
//! Therefore, one must properly account for using co- and contravariant basis vectors. Simple geometries may nonetheless have orthogonal basis vectors.

//! Default (primitive) coordinate systems & model geometries:
//! - [`EuclideanNGeometry`] An `N`-dimensional coordinate system (x_1, ... x_n) (abstract only)
//! - [`SphericalGeometry`] A spherical coordinate system (r, ϕ, θ)
//!
//! Each geometry is associated with a fixed coordinate system state type, which enables the description of time-varying coordatinate systems.
//! The coordinate system state types must be initialized from the coordinate system parameters using an implementation of [`Coordinates::initialize_cs`].

mod euclidean;
mod spherical;
mod util;

pub use euclidean::*;
pub use spherical::*;
pub use util::*;

use crate::methods::optimization::{DampedNewtonConfig, damped_newton};
use crate::tval;
use nalgebra::{
    Const, DMatrix, Dim, RealField, SMatrix, SVector, SVectorView, U1, U3, Vector3, VectorView,
};
use rayon::prelude::*;
use std::iter::Sum;

/// A trait that is shared by all coordinate systems that are used within a Bayesian forward model.
///
/// # Type Parameters
/// - `T`: Numeric field type (e.g., f32 or f64 implementing `RealField`)
/// - `D`: Number of spatial dimensions (const generic)
/// - `P`: Number of coordinate system parameters (const generic)
///
/// # Associated Types
/// - `CSST`: Coordinate system state type (can be `()` for static systems, or custom struct for time-varying)
///
/// # Coordinate Systems
/// This trait abstracts over different coordinate systems (Cartesian, spherical, etc.) and provides methods for:
/// - Basis vectors (contravariant and covariant representations)
/// - Metric tensor determinant (`sqrt_detg`)
/// - Coordinate transformations between internal coordinates (`internal_coordinates`) and external Cartesian (`external_coordinates`)
///
/// # Strides
/// To allow for arbitrary memory layouts many functions take up to four stride generics.
/// - `CRStride`: Stride for the internal coordinates (row)
/// - `CCStride`: Stride for the internal coordinates (column)
/// - `PRStride`: Stride for the parameters (row)
/// - `PCStride`: Stride for the parameters (column)
///
/// In many usecases the rust can infer these types, but they can also be explicitly specified when needed.
///
/// # Examples
/// - `CartesianGeometry`: (x, y, z) with identity basis
/// - `SphericalGeometry`: (r, φ, θ) with curved basis vectors
/// - `LinearGeometry`: (x) in 1D
pub trait Geometry<T, const D: usize, const P: usize>
where
    T: RealField,
{
    /// Coordinate system parameter names.
    const PARAM_NAMES: SVector<&'static str, P>;

    /// Number of dimensions.
    const NDIMS: usize = D;

    /// Number of parameters.
    const NPARAMS: usize = P;

    /// Associated coordinate system state type.
    type CSST: Clone + Default + Send;

    /// Returns the local contravariant basis vectors.
    ///
    /// The default implementation computes the contravariant basis vectors using finite differences
    /// of the `transform_internal_to_external` function with respect to each internal coordinate.
    /// Each basis vector is computed as the derivative ∂x/∂q_i, where x is the external coordinate
    /// and q_i is the i-th internal coordinate.
    ///
    /// Implementations that have analytical expressions for the basis vectors should override
    /// this method for better accuracy and performance.
    fn contravariant_basis<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView<T, Const<D>, CRStride, CCStride>,
        params: &VectorView<T, Const<P>, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<SMatrix<T, D, D>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        // Use a small finite difference step
        let delta_h = T::from_f64(1e-6).unwrap();
        let half_delta = delta_h.clone() / tval!(2, usize);

        let mut basis_vectors = Vec::with_capacity(D);

        for i in 0..D {
            // Create perturbed coordinates: q + e_i * delta_h/2
            let mut coord_plus = internal_coordinates.clone_owned();
            coord_plus[i] += half_delta.clone();

            // Create perturbed coordinates: q - e_i * delta_h/2
            let mut coord_minus = internal_coordinates.clone_owned();
            coord_minus[i] -= half_delta.clone();

            // Evaluate transform at perturbed points
            let external_plus = Self::transform_internal_to_external::<
                U1,
                Const<D>,
                PRStride,
                PCStride,
            >(&coord_plus.as_view(), params, cs_state)?;
            let external_minus = Self::transform_internal_to_external::<
                U1,
                Const<D>,
                PRStride,
                PCStride,
            >(&coord_minus.as_view(), params, cs_state)?;

            // Compute finite difference: ∂x/∂q_i ≈ (x(q + e_i*h/2) - x(q - e_i*h/2)) / h
            let basis_vector = (external_plus - external_minus) / delta_h.clone();
            basis_vectors.push(basis_vector);
        }

        // Construct matrix with basis vectors as columns
        Some(SMatrix::from_columns(&basis_vectors))
    }

    /// Returns the local contravariant basis vectors and returns the normalized vectors.
    fn contravariant_basis_normalized<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView<T, Const<D>, CRStride, CCStride>,
        params: &VectorView<T, Const<P>, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<SMatrix<T, D, D>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        let mut basis = Self::contravariant_basis::<CRStride, CCStride, PRStride, PCStride>(
            internal_coordinates,
            params,
            cs_state,
        )?;

        basis.column_iter_mut().for_each(|mut col| {
            col.set_column(0, &col.normalize());
        });

        Some(basis)
    }

    /// Create a vector from contravariant components.
    fn contravariant_vector<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView<T, Const<D>, CRStride, CCStride>,
        components: &SVectorView<T, D>,
        params: &VectorView<T, Const<P>, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, D>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
        SVector<T, D>: Sum,
    {
        let basis = Self::contravariant_basis::<CRStride, CCStride, PRStride, PCStride>(
            internal_coordinates,
            params,
            cs_state,
        )?;

        Some(
            basis
                .column_iter()
                .zip(components.iter())
                .map(|(basis, component)| basis * component.clone())
                .sum(),
        )
    }

    /// Create a vector from contravariant components, using the normalized basis vectors.
    fn contravariant_vector_normalized<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView<T, Const<D>, CRStride, CCStride>,
        components: &SVectorView<T, D>,
        params: &VectorView<T, Const<P>, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, D>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
        SVector<T, D>: Sum,
    {
        let basis = Self::contravariant_basis_normalized::<CRStride, CCStride, PRStride, PCStride>(
            internal_coordinates,
            params,
            cs_state,
        )?;

        Some(
            basis
                .column_iter()
                .zip(components.iter())
                .map(|(basis, component)| {
                    if basis[0].is_finite() {
                        basis * component.clone()
                    } else {
                        SVector::zeros()
                    }
                })
                .sum(),
        )
    }

    /// Returns the local covariant basis vectors.
    fn covariant_basis<CRStride, CCStride, PRStride, PCStride>(
        _internal_coordinates: &VectorView<T, Const<D>, CRStride, CCStride>,
        _params: &VectorView<T, Const<P>, PRStride, PCStride>,
        _state: &Self::CSST,
    ) -> Option<SMatrix<T, D, D>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        unimplemented!("covariant basis vectors are currently not implemented")
    }

    /// Initialize the coordinate system state type.
    ///
    /// This function may panic for invalid parameters.
    fn initialize_csst<PRStride: Dim, PCStride: Dim>(
        params: &VectorView<T, Const<P>, PRStride, PCStride>,
        cs_state: &mut Self::CSST,
    );

    /// Returns the square root of the determinant of the metric tensor.
    ///
    /// The default implementation computes √det(g) from the contravariant basis vectors.
    /// For a D-dimensional coordinate system, this is computed as the absolute value of
    /// the determinant of the matrix whose columns are the contravariant basis vectors.
    ///
    /// The metric tensor g_ij is related to the basis vectors e_i by:
    /// g_ij = e_i · e_j
    ///
    /// Its determinant can be computed efficiently from the basis vectors.
    ///
    /// Implementations that have analytical expressions for √det(g) should override
    /// this method for better accuracy and performance.
    fn sqrt_detg<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView<T, Const<D>, CRStride, CCStride>,
        params: &VectorView<T, Const<P>, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<T>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        // Get the contravariant basis vectors
        let basis = Self::contravariant_basis::<CRStride, CCStride, PRStride, PCStride>(
            internal_coordinates,
            params,
            cs_state,
        )?;

        // Convert to a dynamic matrix to compute determinant
        let basis_dynamic = DMatrix::from_fn(D, D, |i, j| basis[(i, j)].clone());

        // Compute sqrt(det(g)) = |det(basis)|
        // We use the LU decomposition to compute the determinant
        let lu = basis_dynamic.lu();
        Some(lu.determinant().abs())
    }

    /// Transform internal coords `internal_coordinates` into cartesian coords `external_coordinates`.
    ///
    /// Converts from this geometry's internal coordinate system to external (world/Cartesian) coordinates.
    ///
    /// # Arguments
    /// - `external_coordinates`: External coordinates (Cartesian/world frame)
    /// - `params`: Coordinate system parameters
    /// - `cs_state`: Coordinate system state (for time-varying systems)
    ///
    /// # Returns
    /// Internal coordinates for this geometry, or `None` if transformation fails
    ///
    /// The default trait implementation makes use of a numerical root finding algorithm.
    /// Where possible, it is recommended to implement a closed form solution for the transformation.
    ///
    /// # Strategy
    /// 1. Generate initial guesses distributed evenly throughout [0, 1]^D
    /// 2. Evaluate residual for each guess and pick the best 3
    /// 3. Use damped Newton optimization on each of the 3 best guesses
    /// 4. Return the best converged result, or None if all fail to converge
    fn transform_external_to_internal<CRStride, CCStride, PRStride, PCStride>(
        external_coordinates: &VectorView<T, Const<D>, CRStride, CCStride>,
        params: &VectorView<T, Const<P>, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, D>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        // Generate initial guesses distributed evenly throughout [0, 1]^D
        // Number of guesses per dimension: 2^D for small D, capped at reasonable values
        let guesses_per_dim: usize = match D {
            1 => 11,
            2 => 11,
            3 => 11,
            _ => 9,
        };
        let total_guesses = guesses_per_dim.pow(D as u32);

        // Generate all initial guesses
        let mut initial_guesses = Vec::with_capacity(total_guesses);
        let mut indices = vec![0; D];

        for _ in 0..total_guesses {
            let mut guess = SVector::<T, D>::zeros();
            for d in 0..D {
                let frac = T::from_f64(indices[d] as f64 / (guesses_per_dim - 1) as f64).unwrap();
                guess[d] = frac;
            }
            initial_guesses.push(guess);

            // Increment indices (like counting in base-guesses_per_dim)
            for idx in indices.iter_mut().take(D) {
                *idx += 1;

                if *idx < guesses_per_dim {
                    break;
                }

                *idx = 0;
            }
        }

        // Evaluate residual for each initial guess and collect (residual_norm_sq, guess) pairs
        let external_coords_owned = external_coordinates.clone_owned();
        let mut guess_residuals: Vec<(T, SVector<T, D>)> = initial_guesses
            .into_iter()
            .filter_map(|guess| {
                // Compute residual: transform_internal_to_external(guess) - external_coordinates
                let internal_ext = Self::transform_internal_to_external::<
                    U1,
                    Const<D>,
                    PRStride,
                    PCStride,
                >(&guess.as_view(), params, cs_state)?;
                let residual = internal_ext - &external_coords_owned;
                let residual_norm_sq = residual.norm_squared();
                Some((residual_norm_sq, guess))
            })
            .collect();

        // Sort by residual norm squared and pick the best 3
        guess_residuals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let best_guesses: Vec<SVector<T, D>> = guess_residuals
            .clone()
            .into_iter()
            .take(3)
            .map(|(_, guess)| guess)
            .collect();

        // Configure the damped Newton optimizer
        let config = DampedNewtonConfig::default();

        // Try optimization from each of the 3 best guesses
        let mut best_result: Option<(T, SVector<T, D>)> = None;

        for initial_guess in best_guesses {
            // Define the residual function for this optimization
            let residual_fn = |internal: &SVector<T, D>| -> Option<SVector<T, D>> {
                let external = Self::transform_internal_to_external::<
                    U1,
                    Const<D>,
                    PRStride,
                    PCStride,
                >(&internal.as_view(), params, cs_state)?;
                Some(external - &external_coords_owned)
            };

            // Run damped Newton optimization
            if let Some(opt_result) = damped_newton(residual_fn, initial_guess, &config) {
                // Check if this result converged and is better than previous best
                if opt_result.converged
                    && (best_result.is_none()
                        || opt_result.residual_norm_sq < best_result.as_ref().unwrap().0)
                {
                    best_result = Some((opt_result.residual_norm_sq, opt_result.solution));
                }
            }
        }

        // Return the best converged result, or None if all failed to converge
        best_result.map(|(_, solution)| solution)
    }

    /// Transform external coords `external_coordinates` into the internal coords `internal_coordinates`.
    ///
    /// Converts from external (world/Cartesian) coordinates to this geometry's internal coordinate system.
    ///
    /// # Arguments
    /// - `internal_coordinates`: Internal coordinates (defined by this geometry's coordinate system)
    /// - `params`: Coordinate system parameters
    /// - `cs_state`: Coordinate system state (for time-varying systems)
    ///
    /// # Returns
    /// External Cartesian coordinates, or `None` if transformation fails
    fn transform_internal_to_external<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView<T, Const<D>, CRStride, CCStride>,
        params: &VectorView<T, Const<P>, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, D>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim;
}

/// A trait that is shared by all 3-dimensional coordinate systems describing a model geometry.
pub trait Geometry3D<T, const P: usize>: Geometry<T, 3, P>
where
    T: RealField,
{
    /// Returns three coordinate matrices that describe an iso-surface (for constant mu).
    fn iso_surface_mu<const RAYON_CHUNK_SIZE: usize, RStride: Dim, CStride: Dim>(
        mu: T,
        nu_matrix: DMatrix<T>,
        s_matrix: DMatrix<T>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<[DMatrix<T>; 3]>
    where
        Self::CSST: Sync,
    {
        let vectors = nu_matrix
            .par_column_iter()
            .zip(s_matrix.par_column_iter())
            .chunks(RAYON_CHUNK_SIZE)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .flat_map(|(row_nu, row_s)| {
                        row_nu.iter().zip(row_s.iter()).map(|(nu, s)| {
                            Self::transform_internal_to_external::<U1, U3, RStride, CStride>(
                                &Vector3::from([mu.clone(), nu.clone(), s.clone()]).as_view(),
                                params,
                                cs_state,
                            )
                        })
                    })
                    .collect::<Vec<Option<SVector<T, 3>>>>()
            })
            .collect::<Option<Vec<SVector<T, 3>>>>()?;

        let rc = nu_matrix.nrows();
        let cc = nu_matrix.ncols();

        let xx = DMatrix::from_iterator(rc, cc, vectors.iter().map(|vec| vec.x.clone()));
        let yy = DMatrix::from_iterator(rc, cc, vectors.iter().map(|vec| vec.y.clone()));
        let zz = DMatrix::from_iterator(rc, cc, vectors.iter().map(|vec| vec.z.clone()));

        Some([xx, yy, zz])
    }

    /// Test the trait functions for a specific implementation.
    fn test_implementation<RStride: Dim, CStride: Dim>(
        internal_coordinates: &SVectorView<T, 3>,
        params: &VectorView<T, Const<P>, RStride, CStride>,
        delta_h: T,
    ) where
        Self::CSST: Default,
    {
        use approx::ulps_eq;
        use nalgebra::Vector3;

        let mut cs_state = Self::CSST::default();

        Self::initialize_csst::<RStride, CStride>(params, &mut cs_state);

        let internal_coordinates_1p = internal_coordinates
            + Vector3::<T>::x_axis().into_inner() * delta_h.clone() / tval!(2, usize);
        let internal_coordinates_1m = internal_coordinates
            - Vector3::<T>::x_axis().into_inner() * delta_h.clone() / tval!(2, usize);

        let internal_coordinates_2p = internal_coordinates
            + Vector3::<T>::y_axis().into_inner() * delta_h.clone() / tval!(2, usize);
        let internal_coordinates_2m = internal_coordinates
            - Vector3::<T>::y_axis().into_inner() * delta_h.clone() / tval!(2, usize);

        let internal_coordinates_3p = internal_coordinates
            + Vector3::<T>::z_axis().into_inner() * delta_h.clone() / tval!(2, usize);
        let internal_coordinates_3m = internal_coordinates
            - Vector3::<T>::z_axis().into_inner() * delta_h.clone() / tval!(2, usize);

        let basis = Self::contravariant_basis::<U1, U3, _, _>(
            &internal_coordinates.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let external_coordinates_1p =
            Self::transform_internal_to_external::<U1, U3, RStride, CStride>(
                &internal_coordinates_1p.as_view(),
                &params.rows_generic(0, params.shape_generic().0),
                &cs_state,
            )
            .unwrap();

        let external_coordinates_1m =
            Self::transform_internal_to_external::<U1, U3, RStride, CStride>(
                &internal_coordinates_1m.as_view(),
                &params.rows_generic(0, params.shape_generic().0),
                &cs_state,
            )
            .unwrap();

        let external_coordinates_2p =
            Self::transform_internal_to_external::<U1, U3, RStride, CStride>(
                &internal_coordinates_2p.as_view(),
                &params.rows_generic(0, params.shape_generic().0),
                &cs_state,
            )
            .unwrap();

        let external_coordinates_2m =
            Self::transform_internal_to_external::<U1, U3, RStride, CStride>(
                &internal_coordinates_2m.as_view(),
                &params.rows_generic(0, params.shape_generic().0),
                &cs_state,
            )
            .unwrap();

        let external_coordinates_3p =
            Self::transform_internal_to_external::<U1, U3, RStride, CStride>(
                &internal_coordinates_3p.as_view(),
                &params.rows_generic(0, params.shape_generic().0),
                &cs_state,
            )
            .unwrap();

        let external_coordinates_3m =
            Self::transform_internal_to_external::<U1, U3, RStride, CStride>(
                &internal_coordinates_3m.as_view(),
                &params.rows_generic(0, params.shape_generic().0),
                &cs_state,
            )
            .unwrap();

        assert!(ulps_eq!(
            basis.column(0),
            &((external_coordinates_1p - external_coordinates_1m) / delta_h.clone()).as_view(),
            max_ulps = 5,
            epsilon = tval!(1e-5, f64),
        ));
        assert!(ulps_eq!(
            basis.column(1),
            &((external_coordinates_2p - external_coordinates_2m) / delta_h.clone()).as_view(),
            max_ulps = 5,
            epsilon = tval!(1e-5, f64)
        ));
        assert!(ulps_eq!(
            basis.column(2),
            &((external_coordinates_3p - external_coordinates_3m) / delta_h).as_view(),
            max_ulps = 5,
            epsilon = tval!(1e-5, f64)
        ));

        let sqrtdetg_basis = (basis
            .column(0)
            .cross(&basis.column(1))
            .dot(&basis.column(2)))
        .abs();
        let sqrtdetg_analy = Self::sqrt_detg::<U1, U3, _, _>(
            &internal_coordinates.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        assert!(approx::ulps_eq!(sqrtdetg_basis, sqrtdetg_analy));
    }
}

// Blank implementation for 3D coordinates.
impl<T, OC, const P: usize> Geometry3D<T, P> for OC
where
    T: RealField,
    Self: Geometry<T, 3, P>,
{
}
