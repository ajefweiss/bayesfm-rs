use crate::{
    ConfSeries, EnsembleModel, EnsembleObservations, EnsembleState, ModelError, noise::NullNoise,
    obs::ObsVec,
};
use nalgebra::{
    Const, DMatrix, Dim, Dyn, OMatrix, RealField, SMatrix, SVector, SVectorView, Scalar, U1,
    VectorView,
};
use num_traits::Float;
use prodef::MultivariateNormalDensity;
use rand_distr::{Distribution, StandardNormal};
use std::{iter::Sum, ops::Sub};

/// Generic method that computes the fisher information matrix (FIM) for an observation function that
/// returns vector observables and with a given multivariate normal likelihood for each dimension of the
/// observables.
pub fn fisher_information_matrix<
    M,
    T,
    OC,
    const D: usize,
    const P: usize,
    const N: usize,
    OF,
    RStride: Dim,
    CStride: Dim,
>(
    model: &M,
    obs: (&OC, &ConfSeries<OC>),
    params: &VectorView<T, Const<P>, RStride, CStride>,
    obs_func: &OF,
    likelihood: &MultivariateNormalDensity<T, Dyn>,
) -> Result<SMatrix<T, P, P>, ModelError<T>>
where
    T: Float + RealField + Sum,
    M: EnsembleModel<T, D, P> + Sync,
    OC: Scalar,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
    OF: Fn(&M, &OC, &SVectorView<T, P>, &M::FMST, &M::CSST) -> Result<ObsVec<T, N>, ModelError<T>>
        + Sync,
    StandardNormal: Distribution<T>,
{
    let mut result = SMatrix::<T, P, P>::zeros();

    let mut matrix_pos =
        OMatrix::<T, Const<P>, Dyn>::from_iterator(P, (0..(P * P)).map(|idx| params[idx % P]));
    let mut matrix_neg =
        OMatrix::<T, Const<P>, Dyn>::from_iterator(P, (0..(P * P)).map(|idx| params[idx % P]));

    let mut obs_ensbl_pos =
        EnsembleObservations::new(obs.0.clone(), obs.1.clone(), P, None).unwrap();
    let mut obs_ensbl_neg =
        EnsembleObservations::new(obs.0.clone(), obs.1.clone(), P, None).unwrap();

    let step_sizes =
        SVector::<T, P>::from_iterator(model.prior_domain().size().iter().map(|range| {
            range.unwrap_or(T::zero()) * (T::from_usize(1024).unwrap() * T::epsilon())
        }));

    matrix_pos
        .column_iter_mut()
        .enumerate()
        .for_each(|(pdx, mut column)| {
            column.set_column(0, params);
            column[pdx] += step_sizes[pdx];
        });

    matrix_neg
        .column_iter_mut()
        .enumerate()
        .for_each(|(pdx, mut column)| {
            column.set_column(0, params);
            column[pdx] -= step_sizes[pdx];
        });

    let mut pos = EnsembleState::new(matrix_pos, None, None);
    let mut neg = EnsembleState::new(matrix_neg, None, None);

    model.initialize_states_ensbl(&mut pos)?;
    model.initialize_states_ensbl(&mut neg)?;

    model.simulate_ensbl(
        &mut pos,
        &mut obs_ensbl_pos,
        obs_func,
        &mut None::<&mut NullNoise>,
    )?;

    model.simulate_ensbl(
        &mut neg,
        &mut obs_ensbl_neg,
        obs_func,
        &mut None::<&mut NullNoise>,
    )?;

    let dmus = (0..P)
        .map(|idx| {
            obs_ensbl_pos
                .time_iter()
                .zip(obs_ensbl_neg.time_iter())
                .map(|((.., pos_col), (.., neg_col))| {
                    if step_sizes[idx] == T::zero() {
                        ObsVec::zeros()
                    } else {
                        (&pos_col[idx] - &neg_col[idx])
                            / (T::from_usize(2).unwrap() * step_sizes[idx])
                    }
                })
                .collect::<Vec<ObsVec<T, N>>>()
        })
        .collect::<Vec<Vec<ObsVec<T, N>>>>();

    for cdx in 0..P {
        for rdx in 0..P {
            if cdx <= rdx {
                let dmu_a_mat = DMatrix::from_iterator(
                    N,
                    obs.1.len(),
                    dmus[rdx]
                        .iter()
                        .flat_map(|obsvec| obsvec.into_iter().cloned()),
                );
                let dmu_b_mat = DMatrix::from_iterator(
                    N,
                    obs.1.len(),
                    dmus[cdx]
                        .iter()
                        .flat_map(|obsvec| obsvec.into_iter().cloned()),
                );

                result[(rdx, cdx)] = (0..N)
                    .map(|idx| {
                        likelihood.bilinear_map::<U1, Dyn>(
                            &dmu_a_mat.row(idx).transpose().as_view(),
                            &dmu_b_mat.row(idx).transpose().as_view(),
                        )
                    })
                    .sum::<T>();

                if rdx != cdx {
                    result[(cdx, rdx)] = result[(rdx, cdx)]
                }
            }
        }
    }

    Ok(result)
}
