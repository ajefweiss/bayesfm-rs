use crate::geometry::Geometry;
use nalgebra::{
    ArrayStorage, Dim, RealField, SMatrix, SVector, Scalar, U0, U3, Vector3, VectorView,
    VectorView3,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Spherical coordinate system.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SphericalGeometry<T>(PhantomData<T>)
where
    T: Scalar;

impl<T> Geometry<T, 3, 0> for SphericalGeometry<T>
where
    T: RealField,
{
    const PARAM_NAMES: SVector<&'static str, 0> =
        SVector::from_array_storage(ArrayStorage([[]; 1]));

    type CSST = ();

    fn contravariant_basis<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView<T, U3, CRStride, CCStride>,
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &(),
    ) -> Option<SMatrix<T, 3, 3>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        let r = internal_coordinates[0].clone();
        let phi = internal_coordinates[1].clone();
        let theta = internal_coordinates[2].clone();

        Some(SMatrix::from_columns(&[
            Vector3::new(
                phi.clone().cos() * theta.clone().sin(),
                phi.clone().sin() * theta.clone().sin(),
                theta.clone().cos(),
            ),
            Vector3::new(
                -phi.clone().sin() * theta.clone().sin(),
                phi.clone().cos() * theta.clone().sin(),
                T::zero(),
            ) * r.clone(),
            Vector3::new(
                phi.clone().cos() * theta.clone().cos(),
                phi.sin() * theta.clone().cos(),
                -theta.sin(),
            ) * r,
        ]))
    }

    fn sqrt_detg<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView<T, U3, CRStride, CCStride>,
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &(),
    ) -> Option<T>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        let r = internal_coordinates[0].clone();
        let theta = internal_coordinates[2].clone();

        Some(r.powi(2) * theta.sin())
    }

    fn initialize_csst<PRStride, PCStride>(
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &mut (),
    ) where
        PRStride: Dim,
        PCStride: Dim,
    {
    }

    fn transform_internal_to_external<CRStride, CCStride, PRStride, PCStride>(
        internal_coordinates: &VectorView3<T, CRStride, CCStride>,
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &(),
    ) -> Option<Vector3<T>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        let r = internal_coordinates[0].clone();
        let phi = internal_coordinates[1].clone();
        let theta = internal_coordinates[2].clone();

        Some(Vector3::new(
            r.clone() * phi.clone().cos() * theta.clone().sin(),
            r.clone() * phi.sin() * theta.clone().sin(),
            r * theta.cos(),
        ))
    }

    fn transform_external_to_internal<CRStride, CCStride, PRStride, PCStride>(
        external_coordinates: &VectorView3<T, CRStride, CCStride>,
        _params: &VectorView<T, U0, PRStride, PCStride>,
        _cs_state: &(),
    ) -> Option<Vector3<T>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        let v = external_coordinates;
        let vn = v.norm();

        Some(Vector3::new(
            vn.clone(),
            v[1].clone().atan2(v[0].clone()),
            (v[2].clone() / vn).acos(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry3D;
    use nalgebra::{SVector, U1, U3, Vector3};

    #[test]
    fn test_spherical_coords() {
        let params = SVector::<f64, 0>::from([]);

        SphericalGeometry::initialize_csst(&params.fixed_rows::<0>(0), &mut ());

        let internal_coordinates_ref = Vector3::new(0.56, 0.17, 0.45);

        let external_coordinates =
            SphericalGeometry::transform_internal_to_external::<U1, U3, U1, U0>(
                &internal_coordinates_ref.as_view(),
                &params.fixed_rows::<0>(0),
                &(),
            )
            .unwrap();

        let internal_coordinates_rec =
            SphericalGeometry::transform_external_to_internal::<U1, U3, U1, U0>(
                &external_coordinates.as_view(),
                &params.fixed_rows::<0>(0),
                &(),
            )
            .unwrap();

        assert!((internal_coordinates_rec - internal_coordinates_ref).norm() < 1e-6);

        SphericalGeometry::test_implementation(
            &internal_coordinates_ref.as_view(),
            &params.fixed_rows::<0>(0),
            1e-6,
        );
    }
}
