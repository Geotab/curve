// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use chrono::{DateTime, Duration, Utc};

use geotab_curve::*;

#[derive(Debug, Clone)]
struct PositionSample {
    time: DateTime<Utc>,
    lat: f32,
    lon: f32,
    save: bool,
    valid: bool,
}

impl PositionSample {
    fn new(time: DateTime<Utc>, lat: f32, lon: f32) -> Self {
        Self {
            time,
            lat,
            lon,
            save: false,
            valid: true,
        }
    }
}

impl Sample for PositionSample {
    fn time(&self) -> DateTime<Utc> {
        self.time
    }

    fn set_time(&mut self, time: DateTime<Utc>) {
        self.time = time;
    }
}

impl Valid for PositionSample {
    fn is_valid(&self) -> bool {
        self.valid
    }
}

impl Save for PositionSample {
    fn is_save(&self) -> bool {
        self.save
    }

    fn set_save(&mut self, save: bool) {
        self.save = save;
    }
}

impl Position<f32> for PositionSample {
    fn latitude(&self) -> f32 {
        self.lat
    }

    fn longitude(&self) -> f32 {
        self.lon
    }
}

type SampleCurve = PositionCurve<PositionSample, f32, 5>;

const ALLOWED_ERROR: f32 = 20.0; // in meters

fn main() {
    let values = [
        (43.870_59, -79.289_14),
        (43.870_54, -79.289_16),
        (43.870_36, -79.289_29),
        (43.870_155, -79.289_55),
        (43.869_953, -79.290_01),
        (43.869_797, -79.290_65),
        (43.869_617, -79.291_34),
        (43.869_44, -79.292_03),
        (43.869236, -79.292_465),
        (43.868824, -79.292_88),
        (43.868_366, -79.293_08),
        (43.867_905, -79.293_03),
        (43.867_47, -79.292_854),
        (43.867_035, -79.292_65),
        (43.866_547, -79.292_42),
        (43.866_11, -79.292_24),
        (43.865_83, -79.292_08),
        (43.865_6, -79.291_985),
        (43.865_574, -79.291954),
    ];

    let mut curve = SampleCurve::new();
    let mut saved = vec![];

    println!("Raw points");
    for (i, (lat, lon)) in values.into_iter().enumerate() {
        let time = DateTime::<Utc>::MIN_UTC + Duration::seconds(i as i64);
        let sample = PositionSample::new(time, lat, lon);
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
