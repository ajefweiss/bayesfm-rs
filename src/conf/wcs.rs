use crate::conf::{ConfPosition, ConfTime};
use nalgebra::{Const, Dyn, OMatrix, OVector, RealField, SVector, Scalar, Vector3};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, fmt::Debug, ops::Sub};

use wcs::{ImgXY, LonLat, WCS, WCSParams};

/// A struct for storing a 3D observation configuration with a camera defined by the WCS.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WCSConf<T>
where
    T: Scalar,
{
    timestamp: T,
    position: Vector3<T>,

    /// WCS params object that describes the camera.
    params: WCSParams,
}

impl<T> WCSConf<T>
where
    T: Scalar,
{
    /// Creates a new vectorized configuration.
    pub fn new(timestamp: T, position: Vector3<T>, params: WCSParams) -> Self {
        Self {
            timestamp,
            position,
            params,
        }
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