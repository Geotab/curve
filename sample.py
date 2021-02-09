# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import bufferedCurve

# Sample dataset
input_time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
input_data = [1000, 1000, 3000, 6000, 5000, 4000, 4000,
              1000, 2000, 3000, 2000, 4000, 5000, 6000, 6001]

# Sample curve configurations
BUFFER_SIZE = 7
ALLOWED_ERROR = 15
DATA_RATIO = 100
TIME_RATIO = 100

# Initialize the curve
curve = bufferedCurve.Curve(bufferSize=BUFFER_SIZE, allowedError=ALLOWED_ERROR, dataRatio=DATA_RATIO,
                            timeRatio=TIME_RATIO,  errorType=bufferedCurve.Distance.PERPENDICULAR)

# Iterating over the dataset and adding some points to the curve
for i in range(0, 5):
    curve.add_point(input_time[i], input_data[i])

# Some points were addded, we would like to simpify them and use them, but intend to continue adding points after
# We are reducing the curve, but not force saving last point (that should only be done when no more points are to be added)
curve.reduce_current_buffer(forceSaveLastPoint=False)

# Get the first part of the reduced points from the algorithm, and reset the curve.
# Reseting the curve will reset the buffers.
output_data = curve.get_reduced_points(resetCurve=True)
print(output_data)  # [[0, 1000], [1, 1000], [3, 6000]]

# Adding the rest of the points to the curve
for i in range(5, len(input_data)):
    curve.add_point(input_time[i], input_data[i])

# The last buffer is not filled, so the reduce function did not run.
# We can force the reduction since there are no more points to be added.
# Reduce the points in the buffer, and force save the last point
curve.reduce_current_buffer(forceSaveLastPoint=True)

# Get the second part of the reduced dataset and optionally reset the curve
output_data = curve.get_reduced_points(resetCurve=True)

print(output_data)  # [[5, 4000], [7, 1000], [14, 6001]]
