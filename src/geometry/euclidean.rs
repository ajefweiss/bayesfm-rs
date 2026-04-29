use crate::geometry::Geometry;
use nalgebra::{ArrayStorage, Dim, RealField, SMatrix, SVector, SVectorView, U0, VectorView};
use std::marker::PhantomData;

/// Linear coordinate system.
pub type LinearGeometry<T> = EuclideanNGeometry<T, 1>;

/// Planar coordinate system.
pub type PlanarGeometry<T> = EuclideanNGeometry<T, 2>;

/// Euclidean coordinate system.
pub type EuclideanGeometry<T> = EuclideanNGeometry<T, 3>;

/// Euclidean `N`-dimensional coordinate system.
pub struct EuclideanNGeometry<T, const N: usize>(PhantomData<T>)
where
    T: RealField;

impl<T, const N: usize> Default for EuclideanNGeometry<T, N>
where
    T: RealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T, const N: usize> Geometry<T, N, 0> for EuclideanNGeometry<T, N>
where
    T: RealField,
{
    const PARAM_NAMES: SVector<&'static str, 0> =
        SVector::from_array_storage(ArrayStorage([[]; 1]));

    type CSST = ();

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        _internal_coordinates: &SVectorView<T, N>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &Self::CSST,
    ) -> Option<SMatrix<T, N, N>> {
        Some(SMatrix::from_diagonal_element(T::one()))
    }

    fn sqrt_detg<RStride: Dim, CStride: Dim>(
        _internal_coordinates: &SVectorView<T, N>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &Self::CSST,
    ) -> Option<T> {
        Some(T::one())
    }

    fn initialize_csst<RStride: Dim, CStride: Dim>(
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &mut Self::CSST,
    ) {
    }

    fn transform_internal_to_external<RStride: Dim, CStride: Dim>(
        internal_coordinates: &SVectorView<T, N>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &Self::CSST,
    ) -> Option<SVector<T, N>> {
        Some(internal_coordinates.clone_owned())
    }

    fn transform_external_to_internal<RStride: Dim, CStride: Dim>(
        external_coordinates: &SVectorView<T, N>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &Self::CSST,
    ) -> Option<SVector<T, N>> {
        Some(external_coordinates.clone_owned())
    }
}
