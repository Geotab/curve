import unittest
import bufferedCurve
from os import path


class TestMethods(unittest.TestCase):
    BUFFER_SIZE = 10
    ALLOWED_ERROR = 5
    ERROR_TYPE = bufferedCurve.Distance.VERTICAL

    def setUp(self):
        self.curve = bufferedCurve.Curve(
            self.BUFFER_SIZE, self.ALLOWED_ERROR, errorType=self.ERROR_TYPE)

    def tearDown(self):
        del(self.curve)

    def test_pointAdd(self):
        points_at_start = self.curve.buffer.get_num_points()
        self.curve.add_point(0, 0)
        self.assertEqual(self.curve.buffer.get_num_points(),
                         points_at_start + 1, 'point not added properly')

    def test_fillBuffer(self):
        [self.curve.add_point(0, 0) for _ in range(self.BUFFER_SIZE)]
        self.assertEqual(self.curve.buffer.get_num_points(),
                         self.BUFFER_SIZE-1, 'Buffer not filled properly')
        # It is not possible to have BUFFER_SIZE points in the buffer. Whenever BUFFER_SIZE points are present
        # in the buffer, the buffer will be optimized.

    def test_zeroDistance(self):
        p1 = bufferedCurve.BufferedPoint(0, 0)
        p2 = bufferedCurve.BufferedPoint(0, 0)
        p3 = bufferedCurve.BufferedPoint(0, 0)
        self.assertEqual(self.curve.distance(p1, p2, p3), 0,
                         'Zero distance not calculated properly')

    def test_oneDistance(self):
        p1 = bufferedCurve.BufferedPoint(3, 1)
        p2 = bufferedCurve.BufferedPoint(0, 0)
        p3 = bufferedCurve.BufferedPoint(5, 0)
        self.assertEqual(self.curve.distance(p1, p2, p3), 1,
                         'One distance not calculated properly')

    def test_backwardsOneDistance(self):
        p1 = bufferedCurve.BufferedPoint(3, 1)
        p2 = bufferedCurve.BufferedPoint(5, 0)
        p3 = bufferedCurve.BufferedPoint(0, 0)
        self.assertEqual(self.curve.distance(p1, p2, p3), 1,
                         'One distance (backwards) not calculated properly')

    def test_reduceIterSamePoint(self):
        # If the same point is added multiple times, it should only be kept once
        [self.curve.add_point(0, 0) for _ in range(self.BUFFER_SIZE)]
        self.curve.reduce_current_buffer(False)
        self.assertEqual(len(self.curve.get_reduced_points()),
                         1, 'reduceIter same point - more than one point kept')

    def test_reduceForceSaveLast(self):
        # Testing force save of last point added
        [self.curve.add_point(i, 0) for i in range(self.BUFFER_SIZE)]
        self.curve.reduce_current_buffer(True)
        self.assertEqual(len(self.curve.get_reduced_points()),
                         2, 'reduceIter save last - 2 points not saved')
        self.assertEqual(self.curve.get_reduced_points(
        )[-1], [9, 0], 'reduceIter save last - last point saved is not last added')

    def test_maxSavedPoints(self):
        # The default number of maximum saved points is 500
        # This test checks to ensure the limit is honored
        try:
            # Add oscillating 0,30,0,30 points, this ensures each point gets saved
            [self.curve.add_point(i, 30*(i % 2)) for i in range(510)]
        except:
            pass
        curve_output = self.curve.get_reduced_points(resetCurve=True)
        self.assertEqual(len(curve_output), 500,
                         'More than 500 points saved with default settings')

    def test_dataset1(self):
        input_data = [1, 1, 3, 6, 5, 4, 4, 10, 2, 3, 20, 40, 50, 60, 61]
        for i in range(len(input_data)):
            self.curve.add_point(i, input_data[i])
        self.curve.reduce_current_buffer(forceSaveLastPoint=True)

        expected_output = [[0, 1], [7, 10], [
            9, 3], [11, 40], [13, 60], [14, 61]]
        curve_output = self.curve.get_reduced_points(resetCurve=True)
        self.assertEqual(curve_output, expected_output,
                         'dataset1 unexpected output')

    def test_dataset2(self):
        # curve needs to be redefined with an allowed error of 4 instead of unit test default of 5
        self.curve = bufferedCurve.Curve(
            self.BUFFER_SIZE, 4, errorType=self.ERROR_TYPE)

        input_data = [0, 0, 0, 0, 0, 5, 10, 15, 20, 25, 30, 30, 30, 30,
                      30, 30, 25, 20, 15, 10, 5, 0, 0, 0, 0, 0, 5, 10, 15, 20, 25, 30]
        for i in range(len(input_data)):
            self.curve.add_point(i, input_data[i])
        self.curve.reduce_current_buffer(forceSaveLastPoint=True)

        expected_output = [[0, 0], [4, 0], [10, 30],
                           [15, 30], [21, 0], [25, 0], [31, 30]]
        curve_output = self.curve.get_reduced_points(resetCurve=True)
        self.assertEqual(curve_output, expected_output,
                         'dataset2 unexpected output')

    def test_negativeInputs(self):
        # Testing force save of last point added
        self.curve.add_point(0, 0)
        self.curve.add_point(1, -10)
        self.curve.add_point(2, -20)
        self.curve.add_point(3, 10)
        self.curve.reduce_current_buffer(True)

        expected_output = [[0, 0], [2, -20], [3, 10]]
        curve_output = self.curve.get_reduced_points(resetCurve=True)
        self.assertEqual(curve_output, expected_output,
                         'dataset3 unexpected output')

    def test_filePath(self):
        # Testing the dataset files can be open as expected by the OS
        self.curve = bufferedCurve.Curve(
            self.BUFFER_SIZE, self.ALLOWED_ERROR, errorType=bufferedCurve.Distance.GPS)
        for file_name in ["speedData.csv", "fuelData.csv", "gpsData.csv", "rpmData.csv"]:
            self.assertEqual(path.exists("Data/"+file_name),
                             True, "files cannot be opened")

    def test_verticalDistance(self):
        p1 = bufferedCurve.BufferedPoint(0, 0)
        p2 = bufferedCurve.BufferedPoint(1, 3)
        p3 = bufferedCurve.BufferedPoint(2, 1)
        self.assertEqual(self.curve.vertical_value(
            p2, p1, p3), 2.5, "vertical distance failed")

    def test_perpendicularDistance(self):
        p1 = bufferedCurve.BufferedPoint(0, 0)
        p2 = bufferedCurve.BufferedPoint(1, 3)
        p3 = bufferedCurve.BufferedPoint(2, 1)
        self.assertEqual(self.curve.perpendicular_value(
            p2, p1, p3), 2.23606797749979, "perpendicular distance failed")


if __name__ == '__main__':
    unittest.main()
