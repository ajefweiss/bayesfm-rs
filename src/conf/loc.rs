use crate::conf::{ConfPosition, ConfTime};
use nalgebra::{Const, Dyn, OMatrix, OVector, RealField, SVector, Scalar};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, fmt::Debug, ops::Sub};

/// A basic vectorized configuration type, storing a timestamp and a `D`-dimensional position.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Location<T, const D: usize>
where
    T: Scalar,
{
    timestamp: T,
    position: SVector<T, D>,
}

impl<T, const D: usize> Location<T, D>
where
    T: Scalar,
{
    /// Creates a new vectorized configuration.
    pub fn new(timestamp: T, position: SVector<T, D>) -> Self {
        Self {
            timestamp,
            position,
        }
    }
}

impl<T, const D: usize> ConfPosition<T, D> for Location<T, D>
where
    T: Scalar,
{
    fn position(&self) -> SVector<T, D> {
        self.position.clone()
    }
}

impl<T, const D: usize> ConfTime<T> for Location<T, D>
where
    T: RealField,
{
    fn timestamp(&self) -> T {
        self.timestamp.clone()
    }
}

impl<T, const D: usize> Default for Location<T, D>
where
    T: RealField,
{
    fn default() -> Self {
        Self {
            timestamp: T::zero(),
            position: SVector::zeros(),
        }
    }
}

impl<T, const D: usize> From<(T, OVector<T, Const<D>>)> for Location<T, D>
where
    T: Scalar,
{
    fn from((timestamp, position): (T, OVector<T, Const<D>>)) -> Self {
        Self::new(timestamp, position)
    }
}

impl<T, const D: usize> PartialOrd for Location<T, D>
where
    T: Scalar + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.timestamp.partial_cmp(&other.timestamp)
    }
}

impl<'a, T, const D: usize> Sub<&'a Location<T, D>> for &'a Location<T, D>
where
    T: PartialOrd + Scalar + Sub<Output = T>,
{
    type Output = T;

    fn sub(self, other: Self) -> Self::Output {
        self.timestamp.clone() - other.timestamp.clone()
    }
}

/// A basic list of vectorized configuration types, storing a timestamp and a matrix of `D`-dimensional positions.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LocationList<T, const D: usize>
where
    T: Scalar,
{
    timestamp: T,
    positions: OMatrix<T, Const<D>, Dyn>,
}

impl<T, const D: usize> LocationList<T, D>
where
    T: Scalar,
{
    /// Creates a new vectorized configuration.
    pub fn new(timestamp: T, positions: OMatrix<T, Const<D>, Dyn>) -> Self {
        Self {
            timestamp,
            positions,
        }
    }

    /// Return individual [`Location`]s for each position in the list, with the same timestamp.
    pub fn split(&self) -> Vec<Location<T, D>> {
        self.positions
            .column_iter()
            .map(|pos| Location::new(self.timestamp.clone(), pos.clone_owned()))
            .collect()
    }
}

impl<T, const D: usize> ConfTime<T> for LocationList<T, D>
where
    T: RealField,
{
    fn timestamp(&self) -> T {
        self.timestamp.clone()
    }
}

impl<T, const D: usize> Default for LocationList<T, D>
where
    T: RealField,
{
    fn default() -> Self {
        Self {
            timestamp: T::zero(),
            positions: OMatrix::zeros_generic(Const::<D>, Dyn(1)),
        }
    }
}

impl<T, const D: usize> From<(T, OMatrix<T, Const<D>, Dyn>)> for LocationList<T, D>
where
    T: Scalar,
{
    fn from((timestamp, positions): (T, OMatrix<T, Const<D>, Dyn>)) -> Self {
        Self::new(timestamp, positions)
    }
}

impl<T, const D: usize> PartialOrd for LocationList<T, D>
where
    T: Scalar + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.timestamp.partial_cmp(&other.timestamp)
    }
}

impl<'a, T, const D: usize> Sub<&'a LocationList<T, D>> for &'a LocationList<T, D>
where
    T: PartialOrd + Scalar + Sub<Output = T>,
{
    type Output = T;

    fn sub(self, other: Self) -> Self::Output {
        self.timestamp.clone() - other.timestamp.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConfSeries;
    use nalgebra::Vector3;

    /// Tests combining and uncombining configurations to verify composite index tracking.
    ///
    /// This test validates that when multiple `ConfSeries` instances are combined using the
    /// `+` operator, the resulting `composite_indices` correctly tracks which observer
    /// each configuration belongs to.
    #[test]
    fn test_conf_combine_and_uncombine() {
        let conf1 = ConfSeries::from_iter([
            Location::from((0.0, Vector3::new(1.0, 0.0, 0.0))),
            Location::from((0.1, Vector3::new(2.0, 0.0, 0.0))),
        ]);

        let conf2 = ConfSeries::from_iter([Location::from((0.05, Vector3::new(1.5, 0.0, 0.0)))]);

        // After combining, total length should be 3
        let combined = conf1.clone() + conf2.clone();
        assert_eq!(
            combined.len(),
            3,
            "Combined configuration should have 3 elements"
        );
        assert_eq!(
            combined.count(),
            2,
            "Combined configuration should have 2 observers"
        );

        // Uncombine should split back into 2 separate configurations
        let uncombined = combined.uncombine();
        assert_eq!(
            uncombined.len(),
            2,
            "Uncombined should produce 2 configurations"
        );
        assert_eq!(
            uncombined[0].len(),
            2,
            "First observer should have 2 configurations"
        );
        assert_eq!(
            uncombined[1].len(),
            1,
            "Second observer should have 1 configuration"
        );

        // Verify timestamps are preserved through struct field access
        assert_eq!(
            uncombined[0].first().unwrap().timestamp,
            0.0,
            "First config of observer 0 should have timestamp 0.0"
        );
        assert_eq!(
            uncombined[0].last().unwrap().timestamp,
            0.1,
            "Last config of observer 0 should have timestamp 0.1"
        );
        assert_eq!(
            uncombined[1].first().unwrap().timestamp,
            0.05,
            "First (only) config of observer 1 should have timestamp 0.05"
        );
    }

    /// Tests subset extraction while preserving observer tracking via composite indices.
    ///
    /// When extracting a subset of configurations using indices, the resulting
    /// `ConfSeries` should maintain correct `composite_indices` values corresponding to the
    /// original multi-observer structure.
    #[test]
    fn test_conf_subset() {
        let conf = ConfSeries::from_iter([
            Location::from((0.0, Vector3::new(1.0, 0.0, 0.0))),
            Location::from((1.0, Vector3::new(2.0, 0.0, 0.0))),
            Location::from((0.5, Vector3::new(1.5, 0.0, 0.0))),
            Location::from((2.0, Vector3::new(3.0, 0.0, 0.0))),
        ]);

        // Extract indices 0, 2, 3 (subset of 4 items)
        let subset = conf
            .subset(&[0, 2, 3])
            .expect("valid indices should succeed");
        assert_eq!(subset.len(), 3, "Subset should have 3 elements");

        // Verify timestamps are correct for the subset via direct field access
        assert_eq!(
            subset.configurations()[0].timestamp,
            0.0,
            "First subset element should have timestamp 0.0"
        );
        assert_eq!(
            subset.configurations()[1].timestamp,
            0.5,
            "Second subset element should have timestamp 0.5"
        );
        assert_eq!(
            subset.configurations()[2].timestamp,
            2.0,
            "Third subset element should have timestamp 2.0"
        );

        // Verify positions are preserved
        assert_eq!(
            subset.configurations()[0].position.x,
            1.0,
            "First subset element should have x=1.0"
        );
        assert_eq!(
            subset.configurations()[1].position.x,
            1.5,
            "Second subset element should have x=1.5"
        );
    }

    /// Tests that subset() returns None for out-of-bounds indices.
    #[test]
    fn test_conf_subset_out_of_bounds() {
        let conf = ConfSeries::from_iter([
            Location::from((0.0, Vector3::new(1.0, 0.0, 0.0))),
            Location::from((1.0, Vector3::new(2.0, 0.0, 0.0))),
            Location::from((0.5, Vector3::new(1.5, 0.0, 0.0))),
        ]);

        // Try to access index 5 when only 0-2 exist
        let result = conf.subset(&[0, 5]);
        assert!(
            result.is_none(),
            "subset() should return None for out-of-bounds index"
        );
    }

    /// Tests that subset() returns None for single out-of-bounds index.
    #[test]
    fn test_conf_subset_single_out_of_bounds() {
        let conf = ConfSeries::from_iter([Location::from((0.0, Vector3::new(1.0, 0.0, 0.0)))]);

        // Try to access index 10 when only 0 exists
        let result = conf.subset(&[10]);
        assert!(
            result.is_none(),
            "subset() should return None on index 10 for len=1"
        );
    }

    /// Tests observer counting and edge cases for single and multi-observer configurations.
    ///
    /// This test ensures that `count()` correctly returns the number of unique observers
    /// in both single-observer and multi-observer scenarios.
    #[test]
    fn test_conf_count() {
        // Single observer with 3 configurations
        let single_obs = ConfSeries::from_iter([
            Location::from((0.0, Vector3::new(1.0, 0.0, 0.0))),
            Location::from((1.0, Vector3::new(2.0, 0.0, 0.0))),
            Location::from((0.5, Vector3::new(1.5, 0.0, 0.0))),
        ]);
        assert_eq!(
            single_obs.count(),
            1,
            "Single-observer configuration should report 1 observer"
        );

        // Create multi-observer configuration by combining
        let conf1 = ConfSeries::from_iter([Location::from((0.0, Vector3::new(1.0, 0.0, 0.0)))]);
        let conf2 = ConfSeries::from_iter([Location::from((1.0, Vector3::new(2.0, 0.0, 0.0)))]);
        let conf3 = ConfSeries::from_iter([Location::from((0.5, Vector3::new(1.5, 0.0, 0.0)))]);

        let combined = (conf1 + conf2) + conf3;
        assert_eq!(
            combined.count(),
            3,
            "Combined configuration should report 3 observers"
        );
        assert_eq!(
            combined.len(),
            3,
            "Combined configuration should have 3 total configurations"
        );
    }
}
