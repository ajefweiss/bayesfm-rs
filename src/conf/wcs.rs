use crate::conf::{ConfCamera, ConfPosition, ConfTime};
use nalgebra::{RealField, Scalar, Vector3};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, fmt::Debug, ops::Sub};
use wcs::WCSParams;

/// A struct for storing a 3D observation configuration with a camera defined by a world coordinate system(WCS).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct WCSConf<T>
where
    T: Scalar,
{
    timestamp: T,
    position: Vector3<T>,

    /// WCS params object that describes the camera.
    params: WCSParams,

    /// Additional camera fields.
    flags: Option<Vec<T>>,
}

impl<T> WCSConf<T>
where
    T: Scalar,
{
    /// Creates a new vectorized configuration.
    pub fn new(
        timestamp: T,
        position: Vector3<T>,
        params: WCSParams,
        flags: Option<Vec<T>>,
    ) -> Self {
        Self {
            timestamp,
            position,
            params,
            flags,
        }
    }
}

impl<T> ConfCamera<T> for WCSConf<T>
where
    T: RealField,
{
    fn wcs(&self) -> &WCSParams {
        &self.params
    }
}

impl<T> ConfPosition<T, 3> for WCSConf<T>
where
    T: Scalar,
{
    fn position(&self) -> Vector3<T> {
        self.position.clone()
    }
}

impl<T> ConfTime<T> for WCSConf<T>
where
    T: RealField,
{
    fn timestamp(&self) -> T {
        self.timestamp.clone()
    }
}

impl<T> PartialOrd for WCSConf<T>
where
    T: Scalar + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.timestamp.partial_cmp(&other.timestamp)
    }
}

impl<'a, T> Sub<&'a WCSConf<T>> for &'a WCSConf<T>
where
    T: PartialOrd + Scalar + Sub<Output = T>,
{
    type Output = T;

    fn sub(self, other: Self) -> Self::Output {
        self.timestamp.clone() - other.timestamp.clone()
    }
}
