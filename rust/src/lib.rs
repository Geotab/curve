// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! General curve logic for telemetry samples. Samples are gathered into a curve which is then
//! reduced to remove redundant points.

use std::{iter::Enumerate, marker::PhantomData, ops::Deref};

use arrayvec::ArrayVec;
use chrono::prelude::*;
use num_traits::Float;

/// A data sample that can be curve logged
pub trait Sample {
    fn time(&self) -> DateTime<Utc>;
    fn set_time(&mut self, time: DateTime<Utc>);
}

/// A sample that can be saved
pub trait Save {
    fn is_save(&self) -> bool;
    fn set_save(&mut self, save: bool);
}

/// A sample that may or may not be valid
pub trait Valid {
    fn is_valid(&self) -> bool;
}

const EARTH_RADIUS: f32 = 6371000.0; // in meters

/// Common trait for curve buffers that store samples for curve logging
pub trait Curve<V>
where
    Self: Sized,
    V: Clone + Sample + Save + Valid,
{
    /// Underlying storage type for the curved samples, such as ArrayVec or Vec
    type Buffer: Deref<Target = [V]> + IntoIterator<Item = V>;

    fn new() -> Self;

    fn capacity(&self) -> usize;

    fn len(&self) -> usize {
        self.values().len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    fn values(&self) -> &[V];
    fn values_mut(&mut self) -> &mut [V];
    fn take_values(&mut self) -> Self::Buffer;
    fn push(&mut self, value: V);

    /// Performs an error calculation between 3 samples. These samples can be a scalar with a
    /// timestamp or GPS coordinates depending on the implementation.
    ///
    /// first_sample: First sample in the line.
    /// last_sample: Last sample in the line.
    /// current_sample: A sample from which the error should be calculated.
    ///
    /// Implementations may impose restrictions on what a valid set of 3 samples is. If an invalid set
    /// of samples is provided, implementations return 0.0.
    fn calc_error(first_sample: &V, last_sample: &V, current_sample: &V) -> f32;

    fn add_value(&mut self, mut value: V) {
        value.set_save(false);
        self.push(value);
    }

    fn save_value(&mut self, mut value: V) {
        value.set_save(true);
        self.push(value);
    }

    /// Run the curve algorithm across a segment of the curve buffer, marking the saved points.
    ///
    /// Returns whether any point was saved, the last saved point, and the point with highest error
    /// after the last saved point.
    fn run(
        &mut self,
        allowed_error: f32,
        start_index: usize,
        end_index: usize,
    ) -> (bool, usize, usize) {
        reduce_values(
            &mut self.values_mut()[start_index..end_index],
            allowed_error,
            Self::calc_error,
        )
    }

    /// Run curve algorithm on the entire curve buffer and return an iterator of all the points
    /// that need to be saved
    #[must_use]
    fn reduce(
        &mut self,
        allowed_error: f32,
        run_on: bool,
        save_last: bool,
    ) -> Option<CurveSaveIter<V, Self>> {
        // should this be # of valid values?
        if self.values().len() < 3 {
            return None;
        }

        // reduce_values
        let (mut some_values_saved, mut run_on_start_index, mut run_on_max_error_index) =
            reduce_values(self.values_mut(), allowed_error, Self::calc_error);

        // mark last value to be saved if desired
        if save_last {
            self.values_mut().last_mut().unwrap().set_save(true);
            some_values_saved = true;
            run_on_start_index = self.values().len() - 1;
            run_on_max_error_index = run_on_start_index;
        }

        // save values and initialize new curve
        let values = self.take_values();
        if some_values_saved || run_on {
            Some(CurveSaveIter {
                run_on,
                run_on_start_index,
                run_on_max_error_index,
                curve: self,
                values: values.into_iter().enumerate(),
                phantom: PhantomData,
            })
        } else {
            None
        }
    }
}

/// Iterator that yields saved points from a curve buffer
pub struct CurveSaveIter<'a, V: Clone + Sample + Save + Valid, C: Curve<V>> {
    run_on: bool,
    run_on_start_index: usize,
    run_on_max_error_index: usize,
    curve: &'a mut C,
    values: Enumerate<<<C as Curve<V>>::Buffer as IntoIterator>::IntoIter>,
    phantom: PhantomData<V>,
}

impl<'a, V: Clone + Sample + Save + Valid, C: Curve<V>> Iterator for CurveSaveIter<'a, V, C> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        for (i, value) in self.values.by_ref() {
            // The next curve will consist of the previous curve's last saved point and the highest
            // error point after it
            if self.run_on && (i == self.run_on_start_index || i == self.run_on_max_error_index) {
                self.curve.add_value(value.clone());
            }
            if value.is_save() {
                return Some(value);
            }
        }
        None
    }
}

fn reduce_values<V, F>(values: &mut [V], allowed_error: f32, calc_error: F) -> (bool, usize, usize)
where
    V: Save + Valid,
    F: Fn(&V, &V, &V) -> f32,
{
    // find index of first valid value
    let first_valid_index = values.iter().position(|value| value.is_valid());
    let first_valid_index = if let Some(first_valid_index) = first_valid_index {
        first_valid_index
    } else {
        return (false, 0, values.len() - 1);
    };

    // find index of last valid value
    let last_valid_index = values.iter().rev().position(|value| value.is_valid());
    let last_valid_index = if let Some(mut last_valid_index) = last_valid_index {
        // we iterated in reverse
        last_valid_index = (values.len() - 1) - last_valid_index;
        last_valid_index
    } else {
        return (false, 0, values.len() - 1);
    };

    let mut run_on_start_index = 0;
    let mut run_on_max_error_index = last_valid_index;
    let mut some_values_saved = false;

    // push segment onto processing stack
    let mut stack = vec![(first_valid_index, last_valid_index)];

    // process segments
    while !stack.is_empty() {
        let (start_index, end_index) = stack.pop().unwrap();

        // find maximum error value and corresponding index between start and end values
        let mut max_error = None;
        for index in (start_index + 1)..end_index {
            if values[index].is_valid() {
                let error = calc_error(&values[start_index], &values[end_index], &values[index]);
                if let Some((max_error_value, _)) = max_error {
                    if error > max_error_value {
                        max_error = Some((error, index));
                    }
                } else {
                    max_error = Some((error, index));
                }
            }
        }

        if let Some((max_error_value, max_error_index)) = max_error {
            if max_error_value > allowed_error {
                // mark value to be saved
                values[max_error_index].set_save(true);
                some_values_saved = true;

                // update run-on start index if necessary
                if max_error_index > run_on_start_index {
                    run_on_start_index = max_error_index;
                    run_on_max_error_index = last_valid_index;
                }

                // add left/right segments to processing stack
                stack.push((start_index, max_error_index));
                stack.push((max_error_index, end_index));
            } else {
                // save index of max error value after last saved value
                if some_values_saved && start_index == run_on_start_index {
                    run_on_max_error_index = if max_error_value == 0.0 {
                        last_valid_index
                    } else {
                        max_error_index
                    };
                }
            }
        }
    }

    (
        some_values_saved,
        run_on_start_index,
        run_on_max_error_index,
    )
}

/// A scalar sample that can be curve logged
pub trait ScalarValue<T>: Clone + Sample + Save + Valid {
    fn value(&self) -> T;
}

/// Statically-allocated curve buffer for scalar values
pub struct ScalarValueCurve<S, T, const CAP: usize> {
    values: ArrayVec<S, CAP>,
    phantom: PhantomData<T>,
}

impl<S, T, const CAP: usize> Curve<S> for ScalarValueCurve<S, T, CAP>
where
    S: ScalarValue<T>,
    T: Float + Into<f32>,
{
    type Buffer = ArrayVec<S, CAP>;

    fn capacity(&self) -> usize {
        CAP
    }

    fn new() -> Self {
        Self {
            values: ArrayVec::new(),
            phantom: PhantomData,
        }
    }

    fn values(&self) -> &[S] {
        &self.values
    }

    fn values_mut(&mut self) -> &mut [S] {
        &mut self.values
    }

    fn take_values(&mut self) -> ArrayVec<S, CAP> {
        std::mem::take(&mut self.values)
    }

    fn push(&mut self, value: S) {
        if let Err(e) = self.values.try_push(value) {
            *self.values.last_mut().unwrap() = e.element();
        }
    }

    /// Performs a vertical error calculation between 3 scalars.
    ///
    /// first_scalar: First point in the line.
    /// last_scalar: Last point in the line.
    /// current_scalar: A point from which the error should be calculated.
    ///
    /// If first_scalar > current_scalar > last_scalar time wise, this function returns 0.0.
    fn calc_error(first_scalar: &S, last_scalar: &S, current_scalar: &S) -> f32 {
        if last_scalar.time() <= first_scalar.time() {
            return 0.0;
        }
        if current_scalar.time() <= first_scalar.time() {
            return 0.0;
        }

        let delta_x1 = (last_scalar.time() - first_scalar.time()).num_seconds() as f32;
        let delta_x2 = (current_scalar.time() - first_scalar.time()).num_seconds() as f32;

        let delta_y1 = last_scalar.value() - first_scalar.value();
        let slope = delta_y1.into() / delta_x1;
        let pred_value = first_scalar.value().into() + slope * delta_x2;

        (current_scalar.value().into() - pred_value).abs()
    }
}

/// A GPS position sample that can be curve logged
pub trait Position<T>: Clone + Sample + Save + Valid {
    fn latitude(&self) -> T;
    fn longitude(&self) -> T;
}

/// Statically-allocated curve buffer for position values
pub struct PositionCurve<P, T, const CAP: usize> {
    values: ArrayVec<P, CAP>,
    phantom: PhantomData<T>,
}

impl<P, T, const CAP: usize> Curve<P> for PositionCurve<P, T, CAP>
where
    P: Position<T>,
    T: Float + Into<f32>,
{
    type Buffer = ArrayVec<P, CAP>;

    fn capacity(&self) -> usize {
        CAP
    }

    fn new() -> Self {
        Self {
            values: ArrayVec::new(),
            phantom: PhantomData,
        }
    }

    fn values(&self) -> &[P] {
        &self.values
    }

    fn values_mut(&mut self) -> &mut [P] {
        &mut self.values
    }

    fn take_values(&mut self) -> ArrayVec<P, CAP> {
        std::mem::take(&mut self.values)
    }

    fn push(&mut self, value: P) {
        if let Err(e) = self.values.try_push(value) {
            *self.values.last_mut().unwrap() = e.element();
        }
    }

    /// Performs an error calculation between 3 points.
    ///
    /// first_point: First point in the line.
    /// last_point: Last point in the line.
    /// current_point: A point from which the error should be calculated.
    ///
    /// Returns the error expressed in meters.
    fn calc_error(first_point: &P, last_point: &P, current_point: &P) -> f32 {
        // Determine the angles between vertices from the center of the earth
        let a = est_lat_lon_distance(last_point, current_point);
        let b = est_lat_lon_distance(first_point, current_point);
        let c = est_lat_lon_distance(first_point, last_point);

        // If the first and last points are at the same place, b and c should be the same
        // but send the min anyway
        if c == 0.0 {
            return b.min(a) * EARTH_RADIUS;
        } else if b == 0.0 || a == 0.0 {
            return 0.0;
        }

        //                 current_pt
        //                   / | \
        //                  /  |  \
        //              b  /   |   \    a
        //                /    |h   \
        //               /     |     \
        // first_point  /alpha_|______\ last_point
        //                      c

        // The points are all close enough together that the curvature of the Earth doesn't affect the calculations.
        // Therefore we can use typical trigonometry to find the height of the triangle.
        // a, b, and c are measured in radians, so we will have to adjust to meters in the end.

        // Use cosine law to find interior angle alpha.
        // Sometimes this value is very slightly over 1.0 due to float math
        let cos_alpha = ((b * b + c * c - a * a) / (2.0 * b * c)).clamp(-1.0, 1.0);
        let alpha = cos_alpha.acos();

        // Use the sine law to find the height of the triangle h
        let height = b * alpha.sin();

        // The original python code uses a 1e6 scaling factor
        // https://github.com/Geotab/curve/blob/main/bufferedCurve.py#L309
        // which is based on the 1e7m in 90 degrees of latitute.
        // However, since our height is measured in radians, we use the earth's radius
        height * EARTH_RADIUS
    }
}

// Implements "Equirectangular approximation" distance: https://www.movable-type.co.uk/scripts/latlong.html
fn est_lat_lon_distance<P, T>(start: &P, end: &P) -> f32
where
    P: Position<T>,
    T: Float + Into<f32>,
{
    let start_lat_rad = start.latitude().into().to_radians();
    let end_lat_rad = end.latitude().into().to_radians();
    let start_long_rad = start.longitude().into().to_radians();
    let end_long_rad = end.longitude().into().to_radians();

    let x = (end_long_rad - start_long_rad) * ((start_lat_rad + end_lat_rad) / 2.0).cos();
    let y = end_lat_rad - start_lat_rad;

    (x * x + y * y).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use chrono::{DateTime, Duration, TimeZone, Utc};

    #[derive(Clone, Debug, PartialEq)]
    struct GpsValue {
        latitude: f32,
        longitude: f32,
    }

    #[derive(Clone, Debug)]
    struct GenericSample<V> {
        value: V,
        time: DateTime<Utc>,
        save: bool,
        valid: bool,
    }

    impl<V: Sized> Sample for GenericSample<V> {
        fn time(&self) -> DateTime<Utc> {
            self.time
        }
        fn set_time(&mut self, time: DateTime<Utc>) {
            self.time = time;
        }
    }

    impl<V: Sized> Save for GenericSample<V> {
        fn is_save(&self) -> bool {
            self.save
        }
        fn set_save(&mut self, save: bool) {
            self.save = save;
        }
    }

    impl<V: Sized> Valid for GenericSample<V> {
        fn is_valid(&self) -> bool {
            self.valid
        }
    }

    type ScalarSample = GenericSample<f32>;
    type GpsSample = GenericSample<GpsValue>;

    impl ScalarValue<f32> for ScalarSample {
        fn value(&self) -> f32 {
            self.value
        }
    }

    impl Position<f32> for GpsSample {
        fn latitude(&self) -> f32 {
            self.value.latitude
        }
        fn longitude(&self) -> f32 {
            self.value.longitude
        }
    }

    type TestScalarValueCurve = ScalarValueCurve<ScalarSample, f32, 100>;
    type TestPositionCurve = PositionCurve<GpsSample, f32, 100>;

    impl ScalarSample {
        fn new(time: DateTime<Utc>, value: f32) -> Self {
            Self {
                value,
                time,
                save: false,
                valid: true,
            }
        }
    }

    impl GpsSample {
        const fn new(time: DateTime<Utc>, latitude: f32, longitude: f32) -> Self {
            Self {
                value: GpsValue {
                    latitude,
                    longitude,
                },
                time,
                save: false,
                valid: true,
            }
        }
    }

    #[test]
    fn zero_scalar_error() {
        let zero_scalar = GenericSample::<f32>::new(DateTime::<Utc>::MIN_UTC, 0.0);

        assert_abs_diff_eq!(
            TestScalarValueCurve::calc_error(&zero_scalar, &zero_scalar, &zero_scalar),
            0.0
        );
    }

    #[test]
    fn one_scalar_error() {
        // Note: This replicates a test of the Python implementation:
        // https://github.com/Geotab/curve/blob/main/testCases.py#L38
        // It's important to note that the order of the points is different in this version.
        let time = DateTime::<Utc>::MIN_UTC;
        assert_abs_diff_eq!(
            TestScalarValueCurve::calc_error(
                &GenericSample::<f32>::new(time, 0.0), /* first_point */
                &GenericSample::<f32>::new(time + Duration::seconds(5), 0.0), /* last_point */
                &GenericSample::<f32>::new(time + Duration::seconds(3), 1.0), /* current_point */
            ),
            1.0
        );
    }

    #[test]
    fn vertical_error() {
        // Note: This replicates a test of the Python implementation:
        // https://github.com/Geotab/curve/blob/main/testCases.py#L38
        // It's important to note that the order of the points is different in this version.
        let time = DateTime::<Utc>::MIN_UTC;
        assert_abs_diff_eq!(
            TestScalarValueCurve::calc_error(
                &GenericSample::<f32>::new(time, 0.0), /* first_point */
                &GenericSample::<f32>::new(time + Duration::seconds(2), 1.0), /* last_point */
                &GenericSample::<f32>::new(time + Duration::seconds(1), 3.0), /* current_point */
            ),
            2.5
        );

        assert_abs_diff_eq!(
            TestScalarValueCurve::calc_error(
                &GenericSample::<f32>::new(Utc.timestamp_opt(1591651124, 0).unwrap(), 4.0), /* first_point */
                &GenericSample::<f32>::new(Utc.timestamp_opt(1591651239, 0).unwrap(), 14.0), /* last_point */
                &GenericSample::<f32>::new(Utc.timestamp_opt(1591651128, 0).unwrap(), 0.0), /* current_point */
            ),
            4.347_826
        );
    }

    #[test]
    fn distance() {
        let time = DateTime::<Utc>::MIN_UTC;
        let zero_pos = GenericSample::<GpsValue>::new(time, 0.0, 0.0);
        let one_pos = GenericSample::<GpsValue>::new(time, 1.0, 1.0);
        let two_pos = GenericSample::<GpsValue>::new(time, 2.0, 2.0);

        assert_abs_diff_eq!(est_lat_lon_distance(&zero_pos, &one_pos), 0.024_682_213,);

        assert_abs_diff_eq!(est_lat_lon_distance(&one_pos, &two_pos), 0.024_678_454,);
    }

    #[test]
    fn gps_error() {
        let time = DateTime::<Utc>::MIN_UTC;
        assert_abs_diff_eq!(
            TestPositionCurve::calc_error(
                &GenericSample::<GpsValue>::new(time, 0.0, 0.0), /* first_point */
                &GenericSample::<GpsValue>::new(time, 0.0, 0.0), /* last_point */
                &GenericSample::<GpsValue>::new(time, 0.0, 0.0), /* current_point */
            ),
            0.0
        );

        assert_abs_diff_eq!(
            TestPositionCurve::calc_error(
                &GenericSample::<GpsValue>::new(time, 0.0, 0.0), /* first_point */
                &GenericSample::<GpsValue>::new(time, 2.0, 2.0), /* last_point */
                &GenericSample::<GpsValue>::new(time, 1.0, 1.0), /* current_point */
            ),
            0.0
        );

        assert_abs_diff_eq!(
            TestPositionCurve::calc_error(
                &GenericSample::<GpsValue>::new(time, 0.0, 0.0), /* first_point */
                &GenericSample::<GpsValue>::new(time, 2.0, 2.0), /* last_point */
                &GenericSample::<GpsValue>::new(time, 1.5, 1.0), /* current_point */
            ),
            39_256.336,
        );

        // Test when two base points are on same coordinate, distance is equal to from point to point
        assert_abs_diff_eq!(
            TestPositionCurve::calc_error(
                &GenericSample::<GpsValue>::new(time, 0.0, 0.0), /* first_point */
                &GenericSample::<GpsValue>::new(time, 0.0, 0.0), /* last_point */
                &GenericSample::<GpsValue>::new(time, 1.0, 1.0), /* current_point */
            ),
            157_250.38, /* equal to est_lat_lon_distance((0.0, 0.0), (1.0, 1.0)) */
        );

        // Test when mid point lies on one of the base points
        assert_abs_diff_eq!(
            TestPositionCurve::calc_error(
                &GenericSample::<GpsValue>::new(time, 0.0, 0.0), /* first_point */
                &GenericSample::<GpsValue>::new(time, 1.0, 1.0), /* last_point */
                &GenericSample::<GpsValue>::new(time, 1.0, 1.0), /* current_point */
            ),
            0.0
        );
    }
}
