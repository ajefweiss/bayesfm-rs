//! # Observation configuration types.
//!
//! This module provides abstractions for observer configurations & time-series.
//!
//! ## Examples
//!
//! ### Creating and Combining Configurations
//! ```
//! use bayesfm::conf::ConfSeries;
//!
//! let obs1 = ConfSeries::<i32>::from_iter(vec![0, 1, 2]);
//! let obs2 = ConfSeries::<i32>::from_iter(vec![3, 4]);
//! let combined = obs1.combine(&obs2);
//! assert_eq!(combined.len(), 5);
//! assert_eq!(combined.count(), 2); // 2 observers
//! ```
//!
//! ### Uncombining Composite Configurations
//! ```
//! use bayesfm::conf::ConfSeries;
//!
//! let obs1 = ConfSeries::<i32>::from_iter(vec![0, 1, 2]);
//! let obs2 = ConfSeries::<i32>::from_iter(vec![3, 4]);
//! let combined = obs1.combine(&obs2);
//! let uncombined = combined.uncombine();
//! assert_eq!(uncombined.len(), 2); // 2 ConfSeries objects
//! ```

mod basic;
mod wcs;

pub use wcs::*;
pub use basic::*;

use crate::tval;
use chrono::{DateTime, Datelike, Timelike, Utc};
use derive_more::IntoIterator;
use nalgebra::{RealField, SVector, Scalar};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    fmt::Debug,
    ops::{Add, AddAssign},
};

/// Time-series configuration for one or more observers.
///
/// Represents a sequence of observation times and/or observer configurations.
/// Can be combined with other `ConfSeries` instances to handle multiple observers
/// while tracking which measurements come from which observer.
///
/// # Generic Parameters
/// - `OC`: Observer configuration type (typically a timestamp or configuration struct)
///
/// # Fields
///
/// ## configuration: Vec<OC>
/// The sequence of observer configurations (e.g., timestamps for observations).
/// In the combined case, contains configurations from multiple observers.
///
/// ## composite_indices: Vec<usize>
/// Maps each configuration to its source observer ID in a combined configuration.
///
/// **Semantics:** For combined `ConfSeries`:
/// - Index value `0` → configurations from Observer 0
/// - Index value `1` → configurations from Observer 1
/// - etc.
///
/// **Single observer `ConfSeries`:** All indices are `0` (only one observer)
///
/// # Examples
///
/// ## Single Observer
///
/// ```
/// use bayesfm::conf::ConfSeries;
///
/// let conf = ConfSeries::<i32>::from_iter(vec![0, 1, 2]);
/// // conf.configuration = [0, 1, 2]
/// // conf.composite_indices = [0, 0, 0]  // All from same observer
/// ```
///
/// ## Combining Multiple Observers
///
/// ```
/// use bayesfm::conf::ConfSeries;
///
/// // Observer 0 has observations at times 0, 1
/// let conf_obs0 = ConfSeries::<i32>::from_iter(vec![0, 1]);
///
/// // Observer 1 has observations at times 2, 3, 4
/// let conf_obs1 = ConfSeries::<i32>::from_iter(vec![2, 3, 4]);
///
/// // Combine into single ConfSeries
/// let combined = conf_obs0.combine(&conf_obs1);
///
/// // Result:
/// // combined.configuration = [0, 1, 2, 3, 4]
/// // combined.composite_indices = [0, 0, 1, 1, 1]
/// //                               ^  ^  ↑  ↑  ↑
/// //                           Obs0  Obs0 Obs1...
/// ```
///
/// ## Uncombining Composite Configurations
///
/// ```
/// use bayesfm::conf::ConfSeries;
///
/// let conf_obs0 = ConfSeries::<i32>::from_iter(vec![0, 1]);
/// let conf_obs1 = ConfSeries::<i32>::from_iter(vec![2, 3, 4]);
/// let combined = conf_obs0.combine(&conf_obs1);
/// let uncombined = combined.uncombine();
/// // Returns: [conf_obs0, conf_obs1]  (individual ConfSeries objects)
/// ```
///
/// # Use Cases
///
/// ## Single Observer
/// Filter results from one source:
/// ```
/// use bayesfm::conf::ConfSeries;
///
/// let conf = ConfSeries::<i32>::from_iter(vec![0, 1, 2]);
/// let observations = vec![0.5, 1.5, 2.5];
/// // Process single time-series
/// ```
///
/// ## Multiple Observers
/// Handle data from multiple instruments or locations:
/// ```
/// use bayesfm::conf::ConfSeries;
///
/// let satellite_obs = ConfSeries::<i32>::from_iter(vec![0, 1]);
/// let ground_station_obs = ConfSeries::<i32>::from_iter(vec![2, 3]);
/// let combined = satellite_obs.combine(&ground_station_obs);
/// // combined.composite_indices tells filter which data came from which source
/// ```
///
/// # Relationship to combine() and uncombine()
///
/// - **`combine()`**: Merges two `ConfSeries` objects into one, tracking observer sources in `composite_indices`
/// - **`uncombine()`**: Splits a combined `ConfSeries` back into individual `ConfSeries` objects
/// - **`count()`**: Returns the number of original observers in the combined `ConfSeries`
#[derive(Clone, Debug, Default, Deserialize, IntoIterator, Serialize)]
#[serde(bound(serialize = "OC: Serialize"))]
#[serde(bound(deserialize = "OC: Deserialize<'de>"))]
pub struct ConfSeries<OC> {
    /// Observation configuration time-series (combined if multiple observers).
    ///
    /// For single observers, this is just the sequence of times/configs.
    /// For combined observers, this contains all configurations concatenated.
    #[into_iterator(ref)]
    configuration: Vec<OC>,

    /// Maps each configuration to its source observer ID in composite observations.
    ///
    /// When combining multiple observers with `combine()`, this tracks
    /// which observer each configuration belongs to. Used internally by
    /// filter algorithms to separate observations by source.
    ///
    /// **Value semantics:**
    /// - `0`: Configuration from first observer
    /// - `1`: Configuration from second observer
    /// - etc.
    ///
    /// **For single observers:** All values are `0`
    ///
    /// **For combined observers:** Values range from `0` to `count()-1`
    composite_indices: Vec<usize>,
}

impl<OC> ConfSeries<OC> {
    /// Combines two [`ConfSeries`] objects into a single one.
    pub fn combine(&self, rhs: &Self) -> Self
    where
        OC: Clone,
    {
        let mut configuration = self.configuration.clone();

        configuration.extend(rhs.configuration.clone());

        // Calculate the maximum existing observer index within self.
        let idx_offset = self
            .composite_indices
            .iter()
            .fold(0, |acc: usize, &v| max(acc, v))
            + 1;

        let mut composite_indices = self.composite_indices.clone();

        // Add index_offset to all composite_indices in rhs.
        composite_indices.extend(
            rhs.composite_indices
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );

        Self {
            configuration,
            composite_indices,
        }
    }

    /// Returns the configurations.
    pub fn configurations(&self) -> &[OC] {
        &self.configuration
    }

    /// Returns the number of individual [`ConfSeries`]'s contained within.
    pub fn count(&self) -> usize {
        if self.is_empty() {
            return 0;
        }

        self.composite_indices
            .iter()
            .fold(0, |acc, next| max(acc, *next))
            + 1
    }

    /// Extracts the configurations for a specific observer group.
    ///
    /// Returns a vector of configurations corresponding to the specified group index.
    pub fn extract(&self, group: usize) -> Vec<OC>
    where
        OC: Clone,
    {
        self.configuration
            .iter()
            .zip(&self.composite_indices)
            .filter_map(|(conf, &idx)| if idx == group { Some(conf.clone()) } else { None })
            .collect()  
    }

    /// Extracts the indices of the configurations for a specific observer group.
    pub fn extract_indices(&self, group: usize) -> Vec<usize> {
        self.composite_indices
            .iter()
            .enumerate()
            .filter_map(|(i, &idx)| if idx == group { Some(i) } else { None })
            .collect()  
    }

    /// Returns the first observation configuration, if possible.
    pub fn first(&self) -> Option<&OC> {
        self.configuration.first()
    }

    /// Returns `true`` if the observation contains no elements.
    pub fn is_empty(&self) -> bool {
        self.configuration.is_empty()
    }

    /// Returns the last observation configuration, if possible.
    pub fn last(&self) -> Option<&OC> {
        self.configuration.last()
    }

    /// Returns the number of elements in the observation.
    pub fn len(&self) -> usize {
        self.configuration.len()
    }

    /// Create a new [`ConfSeries`] from a slice of configurations.
    pub fn new(configurations: &[OC]) -> Self
    where
        OC: Clone,
    {
        Self {
            configuration: configurations.to_vec(),
            composite_indices: vec![0; configurations.len()],
        }
    }

    /// Sorts the underlying time-series object by the configurations.
    pub fn sort(&mut self)
    where
        OC: PartialOrd,
    {
        // Implements the bubble sort algorithm for re-ordering all vectors according to the
        // time stamp values.
        let bubble_sort = |configuration: &mut Vec<OC>, composite_indices: &mut Vec<usize>| {
            let mut counter = 0;

            for idx in 0..(configuration.len() - 1) {
                if configuration[idx] > configuration[idx + 1] {
                    configuration.swap(idx, idx + 1);
                    composite_indices.swap(idx, idx + 1);

                    counter += 1
                }
            }

            counter
        };

        let mut counter = 1;

        while counter != 0 {
            counter = bubble_sort(&mut self.configuration, &mut self.composite_indices);
        }
    }

    /// Returns a subset of the observation based on the provided indices.
    ///
    /// Returns `None` if any index is out of bounds. Due to the invariant that
    /// `len(configuration) == len(composite_indices)`, this is the only failure mode.
    ///
    /// # Arguments
    /// * `indices` - Slice of indices to extract (must all be < `self.len()`)
    pub fn subset(&self, indices: &[usize]) -> Option<Self>
    where
        OC: Clone,
    {
        // Validate all indices before processing
        for &i in indices {
            if i >= self.configuration.len() {
                return None;
            }
        }

        let configuration = indices
            .iter()
            .map(|&i| self.configuration[i].clone())
            .collect::<Vec<OC>>();

        let composite_indices = indices
            .iter()
            .map(|&i| self.composite_indices[i])
            .collect::<Vec<usize>>();

        Some(Self {
            configuration,
            composite_indices,
        })
    }

    /// Uncombines the observation into a vector of individual [`ConfSeries`] objects.
    pub fn uncombine(&self) -> Vec<ConfSeries<OC>>
    where
        OC: Clone,
    {
        let count = self.count();
        let mut obs_vec = Vec::with_capacity(count);

        for idx in 0..count {
            let indices = self
                .composite_indices
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v == idx { Some(i) } else { None })
                .collect::<Vec<usize>>();

            // Safe to unwrap: indices are guaranteed valid by the ConfSeries invariant
            obs_vec.push(self.subset(&indices).unwrap());
        }

        obs_vec
    }

    /// Returns the uncombined composite_indices.
    /// 
    /// These indices can be used to re-sort a list or array into the same order as the combined configuration.
    pub fn uncombined_indices(&self) -> Vec<usize> {
        let count = self.count();
        let mut counts = vec![0; count];
        let mut composite_indices = Vec::with_capacity(self.len());

        // Here composite_indices is populated with the index of the original observer for each configuration.
        self.composite_indices.iter().for_each(|&group| {
            composite_indices.push(counts[group]);
            counts[group] += 1;
        });

        composite_indices
            .iter_mut()
            .enumerate()
            .for_each(|(edx, idx)| {
                let mut group = self.composite_indices[edx];

                // Accumulate offsets for all previous groups to get the correct index in the uncombined structure.
                while group > 0 {
                    *idx += counts[group - 1];
                    group -= 1;
                }
            });

        composite_indices
    }
}

impl<OC> Add for ConfSeries<OC>
where
    OC: Clone,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.combine(&rhs)
    }
}

impl<OC> AddAssign for ConfSeries<OC> {
    fn add_assign(&mut self, rhs: Self) {
        self.configuration.extend(rhs.configuration);

        // Calculate the maximum existing observer index within self.
        let idx_offset = self.composite_indices.iter().fold(0, |acc, &v| max(acc, v)) + 1;

        // Add index_offset to all composite_indices in rhs.
        self.composite_indices.extend(
            rhs.composite_indices
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );
    }
}

impl<OC> FromIterator<OC> for ConfSeries<OC> {
    fn from_iter<I: IntoIterator<Item = OC>>(iter: I) -> Self {
        let configuration = iter.into_iter().collect::<Vec<OC>>();
        let length = configuration.len();

        Self {
            configuration,
            composite_indices: vec![0; length],
        }
    }
}

/// A trait for configurations providing positional information.
pub trait ConfPosition<T, const D: usize>
where
    T: Scalar,
{
    /// Returns the position of the configuration.
    fn position(&self) -> SVector<T, D>;
}

/// A trait for configurations providing time information.
pub trait ConfTime<T>: Scalar
where
    T: RealField,
{
    /// Returns the day of the year.
    fn day_of_year(&self) -> usize
    where
        T: AsPrimitive<i64>,
    {
        let datetime =
            DateTime::<Utc>::from_timestamp(self.timestamp().as_(), 0).expect("Invalid timestamp");

        datetime.ordinal() as usize
    }

    /// Returns the local solar time in hours.
    fn local_solar_time(&self, geodetic_longitude: T) -> T
    where
        T: AsPrimitive<i64>,
    {
        // 1. Calculate the day of the year (n)
        let doy = tval!(self.day_of_year(), usize);

        // 3. Fractional year angle in radians
        let gamma = T::two_pi() * ((doy - tval!(1, usize)) / tval!(365, usize));

        // 4. Equation of Time (EoT) in minutes - Spencer (1971) approximation
        // EoT = 229.18 × [0.000075 + 0.001868*cos(γ) - 0.032077*sin(γ)
        //       - 0.014615*cos(2γ) - 0.040849*sin(2γ)]
        let eot_factor = tval!(0.000075, f64) + tval!(0.001868, f64) * gamma.cos()
            - tval!(0.032077, f64) * gamma.sin()
            - tval!(0.014615, f64) * (tval!(2, usize) * gamma).cos()
            - tval!(0.040849, f64) * (tval!(2, usize) * gamma).sin();

        let eot = tval!(229.18, f64) * eot_factor; // in minutes

        // 5. Longitude correction: 4 minutes per degree (longitude West is negative)
        let longitude_correction = tval!(4, usize) * geodetic_longitude;

        // 6. Total time offset in minutes
        let offset_minutes = longitude_correction + eot;

        // 7. UTC time in hours from midnight
        let utc_hours = tval!(self.seconds_of_day(), usize) / tval!(3600, usize);

        // 8. LST in hours (with wraparound to [0, 24))
        let lst = utc_hours + offset_minutes / tval!(60, usize);

        // Ensure LST is in [0, 24)
        let lst_wrapped = lst % tval!(24, usize);
        if lst_wrapped < tval!(0, usize) {
            lst_wrapped + tval!(24, usize)
        } else {
            lst_wrapped
        }
    }

    /// Returns the seconds elapsed since midnight UTC.
    fn seconds_of_day(&self) -> usize
    where
        T: AsPrimitive<i64>,
    {
        let datetime =
            DateTime::<Utc>::from_timestamp(self.timestamp().as_(), 0).expect("Invalid timestamp");

        datetime.num_seconds_from_midnight() as usize
    }

    /// Returns the timestamp of the configuration.
    fn timestamp(&self) -> T;
}
