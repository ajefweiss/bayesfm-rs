use derive_more::From;
use nalgebra::{Matrix, RealField, SMatrix, SVector, SVectorView, Scalar};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, ops::Add};

/// Stores internal coordinates and associated basis vectors for observations.
///
/// This type encapsulates a point in a curvilinear coordinate system along with
/// the basis vectors at that point. It is commonly used in geophysical and
/// heliophysical applications where observations are naturally defined in
/// non-Cartesian coordinate systems (e.g., spherical, cylindrical).
///
/// # Generic Parameters
/// - `T`: Numeric type (typically f32 or f64)
/// - `D`: Number of spatial dimensions
///
/// # Fields
/// - **coordinates**: Internal coordinate values (e.g., (r, φ, θ) in spherical)
///   - Returned by `Geometry::internal()` transformation
///   - Represents position in the curvilinear coordinate system
///
/// - **basis**: Column-major matrix where each column is a basis vector at this point
///   - Dimension: D × D matrix
///   - Column 0: First basis vector (always present, accessed via `eps_mu()`)
///   - Column 1: Second basis vector (3D only, accessed via `eps_nu()`)
///   - Column 2: Third basis vector (3D only, accessed via `eps_s()`)
#[derive(Clone, Debug, Deserialize, From, PartialEq, Serialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
pub struct ObsCoordBasis<T, const D: usize>
where
    T: Scalar,
{
    /// Internal coordinate values in the curvilinear system.
    coordinates: SVector<T, D>,
    /// Column-major matrix of basis vectors at this coordinate point.
    basis: SMatrix<T, D, D>,
}

impl<T, const D: usize> ObsCoordBasis<T, D>
where
    T: RealField,
{
    /// Returns the basis vector matrix (D × D, column-major).
    ///
    /// Each column represents a basis vector at the coordinate point.
    /// For a 3D system, column 0 is eps_mu, column 1 is eps_nu, column 2 is eps_s.
    pub fn basis(&self) -> &SMatrix<T, D, D> {
        &self.basis
    }

    /// Returns the internal coordinates in the curvilinear system.
    ///
    /// These are the coordinates returned by `Geometry::internal()` transformation.
    pub fn coordinates(&self) -> &SVector<T, D> {
        &self.coordinates
    }

    /// Creates a new `ObsCoordBasis` from coordinates and basis matrix.
    pub fn new(coordinates: SVector<T, D>, basis: SMatrix<T, D, D>) -> Self {
        Self { coordinates, basis }
    }

    /// Creates a new `ObsCoordBasis` from coordinates and basis vectors.
    ///
    /// # Parameters
    /// - `coordinates`: Point in the curvilinear coordinate system
    /// - `vectors`: Slice of D basis vectors (each dimension D)
    ///
    /// # Panics
    /// If `vectors.len() != D`.
    ///
    /// # Examples
    /// ```
    /// use bayesfm::obs::ObsCoordBasis;
    /// use nalgebra::SVector;
    ///
    /// let coords: SVector<f64, 3> = SVector::from_row_slice(&[1.0, 0.5, 0.3]);
    /// let e1: SVector<f64, 3> = SVector::from_row_slice(&[1.0, 0.0, 0.0]);
    /// let e2: SVector<f64, 3> = SVector::from_row_slice(&[0.0, 1.0, 0.0]);
    /// let e3: SVector<f64, 3> = SVector::from_row_slice(&[0.0, 0.0, 1.0]);
    ///
    /// let obs = ObsCoordBasis::from_vectors(coords, &[e1, e2, e3]);
    /// assert_eq!(obs.coordinates(), &coords);
    /// ```
    pub fn from_vectors(coordinates: SVector<T, D>, vectors: &[SVector<T, D>]) -> Self {
        Self {
            coordinates,
            basis: Matrix::from_columns(vectors),
        }
    }
}

impl<T> ObsCoordBasis<T, 1>
where
    T: RealField,
{
    /// Returns the first (and only) basis vector (column 0).
    ///
    /// In 1D systems, this is the only basis vector direction.
    /// In multi-dimensional contexts with 1D reduction, this represents the primary direction.
    pub fn eps_mu<'a>(&'a self) -> SVectorView<'a, T, 1> {
        self.basis.column(0)
    }
}

impl<T> ObsCoordBasis<T, 2>
where
    T: RealField,
{
    /// Returns the first basis vector (column 0).
    ///
    /// In 2D systems, this is typically the primary or radial direction.
    /// The semantic meaning depends on the application's coordinate system.
    pub fn eps_mu<'a>(&'a self) -> SVectorView<'a, T, 2> {
        self.basis.column(0)
    }
}

impl<T> ObsCoordBasis<T, 3>
where
    T: RealField,
{
    /// Returns the first basis vector (column 0).
    ///
    /// In 3D systems (e.g., spherical coordinates):
    /// - Common interpretation: Direction of field lines or primary axis
    /// - Alternative: Radial direction in spherical or cylindrical systems
    /// - Semantic meaning is application-defined
    ///
    /// # Examples
    /// ```
    /// use bayesfm::obs::ObsCoordBasis;
    /// use nalgebra::SVector;
    ///
    /// let coords = SVector::from_row_slice(&[1.0, 0.0, 0.0]);
    /// let e1 = SVector::from_row_slice(&[1.0, 0.0, 0.0]);
    /// let e2 = SVector::from_row_slice(&[0.0, 1.0, 0.0]);
    /// let e3 = SVector::from_row_slice(&[0.0, 0.0, 1.0]);
    ///
    /// let obs = ObsCoordBasis::<f64, 3>::from_vectors(coords, &[e1, e2, e3]);
    /// assert_eq!(obs.eps_mu(), e1);
    /// ```
    pub fn eps_mu<'a>(&'a self) -> SVectorView<'a, T, 3> {
        self.basis.column(0)
    }

    /// Returns the second basis vector (column 1).
    ///
    /// Only available for 3D systems. In 3D curvilinear coordinates:
    /// - Common interpretation: First perpendicular direction
    /// - Alternative: Azimuthal direction in spherical/cylindrical systems
    /// - Semantic meaning is application-defined
    ///
    /// # Examples
    /// ```
    /// use bayesfm::obs::ObsCoordBasis;
    /// use nalgebra::SVector;
    ///
    /// let coords = SVector::from_row_slice(&[1.0, 0.0, 0.0]);
    /// let e1 = SVector::from_row_slice(&[1.0, 0.0, 0.0]);
    /// let e2 = SVector::from_row_slice(&[0.0, 1.0, 0.0]);
    /// let e3 = SVector::from_row_slice(&[0.0, 0.0, 1.0]);
    ///
    /// let obs = ObsCoordBasis::<f64, 3>::from_vectors(coords, &[e1, e2, e3]);
    /// assert_eq!(obs.eps_nu(), e2);
    /// ```
    pub fn eps_nu<'a>(&'a self) -> SVectorView<'a, T, 3> {
        self.basis.column(1)
    }

    /// Returns the third basis vector (column 2).
    ///
    /// Only available for 3D systems. Completes the orthogonal basis set.
    /// In 3D curvilinear coordinates:
    /// - Common interpretation: Second perpendicular direction
    /// - Alternative: Polar direction in spherical systems
    /// - Semantic meaning is application-defined
    ///
    /// # Examples
    /// ```
    /// use bayesfm::obs::ObsCoordBasis;
    /// use nalgebra::SVector;
    ///
    /// let coords = SVector::from_row_slice(&[1.0, 0.0, 0.0]);
    /// let e1 = SVector::from_row_slice(&[1.0, 0.0, 0.0]);
    /// let e2 = SVector::from_row_slice(&[0.0, 1.0, 0.0]);
    /// let e3 = SVector::from_row_slice(&[0.0, 0.0, 1.0]);
    ///
    /// let obs = ObsCoordBasis::<f64, 3>::from_vectors(coords, &[e1, e2, e3]);
    /// assert_eq!(obs.eps_s(), e3);
    /// ```
    pub fn eps_s<'a>(&'a self) -> SVectorView<'a, T, 3> {
        self.basis.column(2)
    }
}

impl<T, const D: usize> Add for ObsCoordBasis<T, D>
where
    T: RealField,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let coordinates = self.coordinates + rhs.coordinates;
        let basis = self.basis + rhs.basis;

        Self { coordinates, basis }
    }
}

impl<T, const D: usize> Zero for ObsCoordBasis<T, D>
where
    T: RealField,
{
    fn is_zero(&self) -> bool {
        self.coordinates.is_zero() && self.basis.iter().all(|e| e.is_zero())
    }

    fn set_zero(&mut self) {
        self.coordinates.set_zero();
        for e in &mut self.basis {
            e.set_zero();
        }
    }

    fn zero() -> Self {
        Self {
            coordinates: SVector::zero(),
            basis: SMatrix::zeros(),
        }
    }
}
