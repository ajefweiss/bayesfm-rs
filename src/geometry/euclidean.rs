use crate::geometry::Geometry;
use nalgebra::{ArrayStorage, Const, Dim, RealField, SMatrix, SVector, Scalar, U0, VectorView};
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
    T: Scalar;

impl<T, const N: usize> Default for EuclideanNGeometry<T, N>
where
    T:  RealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T, const N: usize> Geometry<T, N, 0> for EuclideanNGeometry<T, N>
where
    T:  RealField,
{
    const PARAM_NAMES: SVector<&'static str, 0> =
        SVector::from_array_storage(ArrayStorage([[]; 1]));

    type CSST = ();

    fn contravariant_basis<CRStride, CCStride, PRStride, PCStride>(
        _internal_coordinates: &VectorView<T, Const<N>, CRStride, CCStride>,
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &Self::CSST,
    ) -> Option<SMatrix<T, N, N>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        Some(SMatrix::from_diagonal_element(T::one()))
    }

    fn sqrt_detg<CRStride, CCStride, PRStride, PCStride>(
        _internal_coordinates: &VectorView<T, Const<N>, CRStride, CCStride>,
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &Self::CSST,
    ) -> Option<T>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        Some(T::one())
    }

    fn initialize_csst<PRStride: Dim, PCStride: Dim>(
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &mut Self::CSST,
    ) {
    }

    fn transform_internal_to_external<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView<T, Const<N>, CRStride, CCStride>,
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &Self::CSST,
    ) -> Option<SVector<T, N>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        Some(internal_coordinates.clone_owned())
    }

    fn transform_external_to_internal<CRStride, CCStride, PRStride, PCStride>(
        external_coordinates: &VectorView<T, Const<N>, CRStride, CCStride>,
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &Self::CSST,
    ) -> Option<SVector<T, N>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        Some(external_coordinates.clone_owned())
    }
}
