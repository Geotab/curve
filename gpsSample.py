# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import bufferedCurve

# load the GPS data from the file
latitude = []
longitude = []
with open("Data/gpsData.csv") as fp:
    line = fp.readline()  # file header
    while line:
        line = fp.readline()
        if len(line) > 0:
            time, lat, lon = line.split(",")
            latitude.append(float(lat))
            longitude.append(float(lon))

fp.close()


BUFFER_SIZE = 10
ALLOWED_ERROR = 250  # meters
curve_gps = bufferedCurve.Curve(
    BUFFER_SIZE, ALLOWED_ERROR, errorType=bufferedCurve.Distance.GPS, runOnLogic=True)

for lat, lon in zip(latitude, longitude):
    curve_gps.add_point(lat, lon)

curve_gps.reduce_current_buffer(True)
reduced_points = curve_gps.get_reduced_points(True)

print("Number of points kept by algorithm:", len(
    reduced_points), "out of", len(latitude))
#  Number of points kept by algorithm: 15 out of 625
