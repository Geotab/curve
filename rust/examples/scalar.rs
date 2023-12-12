// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use chrono::{DateTime, Duration, Utc};

use geotab_curve::*;

#[derive(Debug, Clone)]
struct ScalarSample {
    time: DateTime<Utc>,
    value: f32,
    save: bool,
    valid: bool,
}

impl ScalarSample {
    fn new(time: DateTime<Utc>, value: f32) -> Self {
        Self {
            time,
            value,
            save: false,
            valid: true,
        }
    }
}

impl Sample for ScalarSample {
    fn time(&self) -> DateTime<Utc> {
        self.time
    }

    fn set_time(&mut self, time: DateTime<Utc>) {
        self.time = time;
    }
}

impl Valid for ScalarSample {
    fn is_valid(&self) -> bool {
        self.valid
    }
}

impl Save for ScalarSample {
    fn is_save(&self) -> bool {
        self.save
    }

    fn set_save(&mut self, save: bool) {
        self.save = save;
    }
}

impl ScalarValue<f32> for ScalarSample {
    fn value(&self) -> f32 {
        self.value
    }
}

type SampleCurve = ScalarValueCurve<ScalarSample, f32, 7>;

const ALLOWED_ERROR: f32 = 7.0;

fn main() {
    let values = [5.88, 6.67, 6.27, 7.84, 20.0, 30.0, 6.67, 6.27, 7.84, 20.0];
    let mut curve = SampleCurve::new();
    let mut saved = vec![];

    println!("Raw points");
    for (i, val) in values.into_iter().enumerate() {
        let time = DateTime::<Utc>::MIN_UTC + Duration::seconds(i as i64);
        let sample = ScalarSample::new(time, val);
        println!("{sample:?}");
        curve.add_value(sample);

        // Reduce the curve buffer when it gets full. Don't force save last point, but make sure to
        // populate the next curve with unsaved curves from the reduced curve (run_on = true)
        if curve.is_full() {
            let reduced = curve.reduce(ALLOWED_ERROR, true, false).unwrap();
            saved.extend(reduced);
            println!("Reduce curve");
        }
    }
    // Reduce the curve one last time, saving the last point
    let reduced = curve.reduce(ALLOWED_ERROR, false, true).unwrap();
    saved.extend(reduced);
    println!("Reduce curve last time");

    println!("\nCurve output:");
    for sample in saved {
        println!("{sample:?}");
    }
}
