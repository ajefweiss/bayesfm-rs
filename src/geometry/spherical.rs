use crate::geometry::Geometry;
use nalgebra::{
    ArrayStorage, Dim, RealField, SMatrix, SVector, U0, Vector3, VectorView, VectorView3,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Spherical coordinate system.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SphericalGeometry<T>(PhantomData<T>)
where
    T: RealField;

impl<T> Geometry<T, 3, 0> for SphericalGeometry<T>
where
    T: RealField,
{
    const PARAM_NAMES: SVector<&'static str, 0> =
        SVector::from_array_storage(ArrayStorage([[]; 1]));

    type CSST = ();

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        internal_coordinates: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<SMatrix<T, 3, 3>> {
        let r = internal_coordinates[0].clone();
        let phi = internal_coordinates[1].clone();
        let theta = internal_coordinates[2].clone();

        Some(SMatrix::from_columns(&[
            Vector3::new(
                phi.clone().cos() * theta.clone().clone().sin(),
                phi.clone().sin() * theta.clone().clone().sin(),
                theta.clone().cos(),
            ),
            Vector3::new(
                -phi.clone().sin() * theta.clone().clone().sin(),
                phi.clone().cos() * theta.clone().clone().sin(),
                T::zero(),
            ) * r.clone(),
            Vector3::new(
                phi.clone().cos() * theta.clone().cos(),
                phi.clone().sin() * theta.clone().cos(),
                -theta.clone().clone().sin(),
            ) * r,
        ]))
    }

    fn sqrt_detg<RStride: Dim, CStride: Dim>(
        internal_coordinates: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<T> {
        let r = internal_coordinates[0].clone();
        let theta = internal_coordinates[2].clone();

        Some(r.powi(2) * theta.clone().clone().sin())
    }

    fn initialize_csst<RStride: Dim, CStride: Dim>(
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &mut (),
    ) {
    }

    fn transform_internal_to_external<RStride: Dim, CStride: Dim>(
        internal_coordinates: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<Vector3<T>> {
        let r = internal_coordinates[0].clone();
        let phi = internal_coordinates[1].clone();
        let theta = internal_coordinates[2].clone();

        Some(Vector3::new(
            r.clone() * phi.clone().cos() * theta.clone().clone().sin(),
            r.clone() * phi.clone().sin() * theta.clone().clone().sin(),
            r * theta.clone().cos(),
        ))
    }

    fn transform_external_to_internal<RStride: Dim, CStride: Dim>(
        external_coordinates: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<Vector3<T>> {
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
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_spherical_coords() {
        let params = SVector::<f64, 0>::from([]);

        SphericalGeometry::initialize_csst(&params.fixed_rows::<0>(0), &mut ());

        let internal_coordinates_ref = Vector3::new(0.56, 0.17, 0.45);

        let external_coordinates = SphericalGeometry::transform_internal_to_external(
            &internal_coordinates_ref.as_view(),
            &params.fixed_rows::<0>(0),
            &(),
        )
        .unwrap();

        let internal_coordinates_rec = SphericalGeometry::transform_external_to_internal(
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
