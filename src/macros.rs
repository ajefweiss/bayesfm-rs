#![allow(unused)]

/// Concatenate the parameter names of the coordinates and the model parameters.
#[macro_export]
macro_rules! model_impl_concat_strs {
    ($coord_param_names: expr, $model_param_names: expr) => {{
        const LEN_A: usize = $coord_param_names.data.0[0].len();

        let mut param_names = [$model_param_names[0]; LEN_A + $model_param_names.len()];

        let mut i1 = 0;
        let mut i2 = 0;

        while i1 < LEN_A {
            param_names[i1] = $coord_param_names.data.0[0][i1];

            i1 += 1;
        }

        while i2 < $model_param_names.len() {
            param_names[LEN_A + i2] = $model_param_names[i2];

            i2 += 1;
        }

        nalgebra::ArrayStorage([param_names; 1])
    }};
}

/// Re-implement the [`$crate::geometry::Geometry`] trait because we have no inheritance.
#[macro_export]
macro_rules! model_impl_coords {
    ($model: ident, $($coords: tt)::+, $params: expr) => {
        impl<T, G>
            bayesfm::geometry::Geometry<
                T,
                { $($coords)::+::<f64>::NDIMS },
                { $($coords)::+::<f64>::NPARAMS + $params.len() },
            > for $model<T, G>
        where
            T:  nalgebra::RealField,
        {
            const PARAM_NAMES: nalgebra::SVector<
                &'static str,
                { $($coords)::+::<f64>::NPARAMS + $params.len() },
            > = nalgebra::SVector::from_array_storage(bayesfm::model_impl_concat_strs!(
                $($coords)::+::<f64>::PARAM_NAMES,
                $params
            ));

            type CSST = <$($coords)::+<T> as bayesfm::geometry::Geometry<
                T,
                { $($coords)::+::<f64>::NDIMS },
                { $($coords)::+::<f64>::NPARAMS },
            >>::CSST;

            fn contravariant_basis<CRStride, CCStride, PRStride, PCStride>(
                internal_coordinates: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $($coords)::+::<f64>::NDIMS }>,
                    CRStride,
                    CCStride
                >,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $($coords)::+::<f64>::NPARAMS + $params.len() }>,
                    PRStride,
                    PCStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<
                nalgebra::SMatrix<
                    T,
                    { $($coords)::+::<f64>::NDIMS },
                    { $($coords)::+::<f64>::NDIMS },
                >,
            >
            where
                CRStride: nalgebra::Dim,
                CCStride: nalgebra::Dim,
                PRStride: nalgebra::Dim,
                PCStride: nalgebra::Dim,
            {
                $($coords)::+::<T>::contravariant_basis(
                    internal_coordinates,
                    &params.fixed_rows::<{ $($coords)::+::<f64>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn sqrt_detg<CRStride, CCStride, PRStride, PCStride>(
                internal_coordinates: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $($coords)::+::<f64>::NDIMS }>,
                    CRStride,
                    CCStride
                >,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $($coords)::+::<f64>::NPARAMS + $params.len() }>,
                    PRStride,
                    PCStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<T>
            where
                CRStride: nalgebra::Dim,
                CCStride: nalgebra::Dim,
                PRStride: nalgebra::Dim,
                PCStride: nalgebra::Dim,
            {
                $($coords)::+::<T>::sqrt_detg(
                    internal_coordinates,
                    &params.fixed_rows::<{ $($coords)::+::<f64>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn initialize_csst<PRStride: nalgebra::Dim, PCStride: nalgebra::Dim>(
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $($coords)::+::<f64>::NPARAMS + $params.len() }>,
                    PRStride,
                    PCStride,
                >,
                cs_state: &mut Self::CSST,
            ) {
                $($coords)::+::<T>::initialize_csst(
                    &params.fixed_rows::<{ $($coords)::+::<f64>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn transform_external_to_internal<CRStride, CCStride, PRStride, PCStride>(
                external_coordinates: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $($coords)::+::<f64>::NDIMS }>,
                    CRStride,
                    CCStride
                >,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $($coords)::+::<f64>::NPARAMS + $params.len() }>,
                    PRStride,
                    PCStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<nalgebra::SVector<T, { $($coords)::+::<f64>::NDIMS }>>
            where
                CRStride: nalgebra::Dim,
                CCStride: nalgebra::Dim,
                PRStride: nalgebra::Dim,
                PCStride: nalgebra::Dim,
            {
                $($coords)::+::<T>::transform_external_to_internal::<CRStride, CCStride, PRStride, PCStride>(
                    external_coordinates,
                    &params.fixed_rows::<{ $($coords)::+::<f64>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn transform_internal_to_external<CRStride, CCStride, PRStride, PCStride>(
                internal_coordinates: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $($coords)::+::<f64>::NDIMS }>,
                    CRStride,
                    CCStride
                >,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $($coords)::+::<f64>::NPARAMS + $params.len() }>,
                    PRStride,
                    PCStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<nalgebra::SVector<T, { $($coords)::+::<f64>::NDIMS }>>
            where
                CRStride: nalgebra::Dim,
                CCStride: nalgebra::Dim,
                PRStride: nalgebra::Dim,
                PCStride: nalgebra::Dim,
            {
                $($coords)::+::<T>::transform_internal_to_external::<CRStride, CCStride, PRStride, PCStride>(
                    internal_coordinates,
                    &params.fixed_rows::<{ $($coords)::+::<f64>::NPARAMS }>(0),
                    cs_state,
                )
            }
        }
    };
}

/// Converts a value to `T`.
#[macro_export]
macro_rules! tval {
    ($expr: expr, usize) => {
        T::from_usize($expr).unwrap()
    };
    ($expr: expr, f64) => {
        T::from_f64($expr).unwrap()
    };
}

pub use {model_impl_concat_strs, model_impl_coords, tval};
