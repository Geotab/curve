// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{
    convert::AsRef,
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use chrono::{prelude::*, Duration};

use geotab_curve::*;

#[derive(Clone, Debug, PartialEq)]
struct SpeedSample {
    time: DateTime<Utc>,
    value: f32,
}

impl Sample for SpeedSample {
    fn time(&self) -> DateTime<Utc> {
        self.time
    }
    fn set_time(&mut self, time: DateTime<Utc>) {
        self.time = time;
    }
}

impl From<String> for SpeedSample {
    fn from(line: String) -> Self {
        let mut fields = line.split(',');
        let field = fields.next().expect("Missing timestamp");
        let timestamp = field.parse().unwrap();
        let time = Utc.timestamp_opt(timestamp, 0).unwrap();
        let field = fields.next().expect("Missing speed");
        let value = field.parse().unwrap();
        Self { time, value }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct GpsSample {
    time: DateTime<Utc>,
    latitude: f32,
    longitude: f32,
}

impl Sample for GpsSample {
    fn time(&self) -> DateTime<Utc> {
        self.time
    }
    fn set_time(&mut self, time: DateTime<Utc>) {
        self.time = time;
    }
}

impl From<String> for GpsSample {
    fn from(line: String) -> Self {
        let mut fields = line.split(',');
        let time = DateTime::<Utc>::MIN_UTC;
        let field = fields.next().expect("Missing latitude");
        let latitude = field.parse().unwrap();
        let field = fields.next().expect("Missing longitude");
        let longitude = field.parse().unwrap();
        Self {
            time,
            latitude,
            longitude,
        }
    }
}

#[derive(Clone, Debug)]
struct CurveSample<S: Sample> {
    line_num: usize,
    sample: S,
    save: bool,
    valid: bool,
}

impl<S: Sample + Sized> From<CurveSample<S>> for (usize, S) {
    fn from(val: CurveSample<S>) -> Self {
        (val.line_num, val.sample)
    }
}

impl<S: Sample + Sized> Save for CurveSample<S> {
    fn is_save(&self) -> bool {
        self.save
    }
    fn set_save(&mut self, save: bool) {
        self.save = save;
    }
}

impl<S: Sample + Sized> Valid for CurveSample<S> {
    fn is_valid(&self) -> bool {
        self.valid
    }
}

type SpeedCurveSample = CurveSample<SpeedSample>;

impl SpeedCurveSample {
    fn new(num: usize, value: f32) -> Self {
        CurveSample {
            line_num: num,
            sample: SpeedSample {
                time: DateTime::<Utc>::MIN_UTC + Duration::seconds(num as i64),
                value,
            },
            save: false,
            valid: true,
        }
    }
}

impl Sample for SpeedCurveSample {
    fn time(&self) -> DateTime<Utc> {
        self.sample.time()
    }
    fn set_time(&mut self, time: DateTime<Utc>) {
        self.sample.set_time(time);
    }
}

impl ScalarValue<f32> for SpeedCurveSample {
    fn value(&self) -> f32 {
        self.sample.value
    }
}

type GpsCurveSample = CurveSample<GpsSample>;

impl Sample for GpsCurveSample {
    fn time(&self) -> DateTime<Utc> {
        self.sample.time()
    }
    fn set_time(&mut self, time: DateTime<Utc>) {
        self.sample.set_time(time);
    }
}

impl Position<f32> for GpsCurveSample {
    fn latitude(&self) -> f32 {
        self.sample.latitude
    }
    fn longitude(&self) -> f32 {
        self.sample.longitude
    }
}

const TEST_CAPACITY: usize = 10;
type SpeedCurve = ScalarValueCurve<SpeedCurveSample, f32, TEST_CAPACITY>;

#[test]
fn save_none() {
    let mut curve = SpeedCurve::new();
    for pt in (0..TEST_CAPACITY).map(|i| SpeedCurveSample::new(i, 3.45)) {
        curve.add_value(pt);
    }
    let saved: Vec<_> = curve.reduce(1.2, true, false).unwrap().collect();
    assert!(saved.is_empty());
    assert_eq!(curve.len(), 2);
    assert_eq!(curve.values()[0].line_num, 0);
    assert_eq!(curve.values()[1].line_num, TEST_CAPACITY - 1);
}

#[test]
fn run_on() {
    let pts = [
        SpeedCurveSample::new(0, 1.23),
        SpeedCurveSample::new(1, 1.83),
        SpeedCurveSample::new(2, 3.45),
        SpeedCurveSample::new(3, 2.5),
        SpeedCurveSample::new(4, 2.23),
    ];

    let mut curve = SpeedCurve::new();
    for pt in &pts {
        curve.add_value(pt.clone());
    }
    // 2 is saved, 2 and 3 are added to next curve as run-on points
    let saved: Vec<_> = curve
        .reduce(1.0, true, false)
        .unwrap()
        .map(|s| s.line_num)
        .collect();
    assert_eq!(saved, vec![2]);
    assert_eq!(curve.len(), 2);
    assert_eq!(curve.values()[0].line_num, 2);
    assert_eq!(curve.values()[1].line_num, 3);

    curve.take_values();
    for pt in pts {
        curve.add_value(pt);
    }
    // 2 and 4 are saved (we're saving last point), 4 is added to next curve
    let saved: Vec<_> = curve
        .reduce(1.0, true, true)
        .unwrap()
        .map(|s| s.line_num)
        .collect();
    assert_eq!(saved, vec![2, 4]);
    assert_eq!(curve.len(), 1);
    assert_eq!(curve.values()[0].line_num, 4);
}

fn print_first_difference<G: Debug + PartialEq>(a: &[(usize, G)], b: &[(usize, G)]) {
    a.iter()
        .zip(b.iter())
        .filter(|(a_entry, b_entry)| a_entry != b_entry)
        .take(1)
        .for_each(|(a_entry, b_entry)| {
            println!("First difference: {:?} != {:?}", a_entry, b_entry)
        });
}

fn assert_check_curve<C, S, P>(mut curve: C, csv_file: P, allowed_error: f32)
where
    C: Curve<CurveSample<S>>,
    CurveSample<S>: Sample,
    S: Clone + Debug + PartialEq + Sample + From<String>,
    P: AsRef<Path>,
{
    let mut expect_saved_values = Vec::new();
    let mut saved_values = Vec::new();

    let mut first_point = true;
    let file = File::open(csv_file).unwrap();
    let reader = BufReader::new(file);
    for (line_num, line_result) in reader.lines().enumerate().skip(1) {
        let line = line_result.unwrap();
        let sample = S::from(line.clone());
        let curve_sample = CurveSample {
            line_num: line_num + 1,
            sample: sample.clone(),
            save: false,
            valid: true,
        };
        if first_point {
            curve.save_value(curve_sample);
            first_point = false;
        } else {
            curve.add_value(curve_sample);
        }

        if let Some(field) = line.split(',').last() {
            if field.to_ascii_lowercase().starts_with('y') {
                expect_saved_values.push((line_num + 1, sample));
            }
        }

        if curve.len() >= curve.capacity() {
            let saved = curve.reduce(allowed_error, true, false);
            saved_values.extend(saved.into_iter().flatten().map(CurveSample::into));
        }
    }

    let saved = curve.reduce(allowed_error, true, true);
    saved_values.extend(saved.into_iter().flatten().map(CurveSample::into));

    println!(
        "Saved {} values, expected {}",
        saved_values.len(),
        expect_saved_values.len()
    );
    print_first_difference(&expect_saved_values, &saved_values);
    assert_eq!(expect_saved_values, saved_values);
}

#[test]
fn speed() {
    assert_check_curve::<SpeedCurve, SpeedSample, &str>(
        SpeedCurve::new(),
        "tests/data/speed.csv",
        4.0,
    );
}
