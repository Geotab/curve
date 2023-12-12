# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math
import copy
from enum import Enum, auto, unique


class BufferedPoint:
    """An point class, with a keep/discard flag."""

    def __init__(self, dateTime: float, value: float, keep: bool = False) -> None:
        self.dateTime = dateTime
        self.value = value
        self.keep = keep


class CurveBuffer:
    def __init__(self, size: int) -> None:
        self.points = []
        self.size = size

    def add(self, point: BufferedPoint) -> None:
        if not self.is_full():
            self.points.append(point)
        else:
            print("Buffer full. Reject new point")

    def clear(self) -> None:
        self.points = []

    def is_full(self) -> bool:
        return len(self.points) >= self.size

    def get_num_points(self) -> int:
        return len(self.points)


@unique
class Distance(Enum):
    VERTICAL = auto()
    PERPENDICULAR = auto()
    GPS = auto()


class Curve:

    def __init__(self, bufferSize: int = 10, allowedError: float = 5, dataRatio: float = 1, timeRatio: float = 1, errorType: Enum = Distance.PERPENDICULAR, forceSaveLastPoint: bool = False, runOnLogic: bool = True, forceSaveZero: bool = False, maxSavedPoints: int = 500) -> None:
        self.loggedPoints = []
        self.buffer = CurveBuffer(bufferSize)
        self.bufferSize = bufferSize
        self.allowedError = allowedError
        self.dataRatio = dataRatio
        self.timeRatio = timeRatio
        self.errorType = errorType
        self.forceSaveLastPoint = forceSaveLastPoint
        self.forceSaveZero = forceSaveZero
        self.runOnLogic = runOnLogic
        self.setCurveToBeReset = False
        self.maxSavedPoints = maxSavedPoints

    def log_points(self) -> int:
        """ Goes over the points in the buffer, and any points that are marked to be kept,
            will be added to the a main buffer that represents the whole curve, and the
            smaller buffer will be cleared.

        Returns:
            int: index of the last added point
        """
        indexOfLastAddedPoint = 0
        for i in range(1, self.buffer.get_num_points()):
            if self.buffer.points[i].keep is True:
                self.add_logged_point(self.buffer.points[i])
                indexOfLastAddedPoint = i

        self.buffer.clear()
        return indexOfLastAddedPoint

    def add_logged_point(self, bufferedPoint: BufferedPoint) -> bool:
        if len(self.loggedPoints) >= self.maxSavedPoints:
            raise (
                "Reduced points buffer is full. Call get_reduced_points(True) to get the points and reset the buffer")

        self.loggedPoints.append([bufferedPoint.dateTime, bufferedPoint.value])
        return True

    def get_point_with_highest_error(self, firstPoint: BufferedPoint, lastPoint: BufferedPoint, buffer: BufferedPoint) -> int:
        maxDistance = 0
        maxDistanceIndex = self.buffer.get_num_points() - 1
        # This might lead to hard to debug behaviour. If the buffer is empty, the value returned by get_num_points is 0, doing -1 points to last point in the buffer (in python)
        # this is the expected behaviour, but it might not be intuitive.

        # skips iterating over first points and last point since those are already saved (so they will have 0 distance)
        for i in range(firstPoint + 1, lastPoint):
            d = self.distance(
                buffer.points[i], buffer.points[firstPoint], buffer.points[lastPoint])
            if d > maxDistance:
                maxDistance = d
                maxDistanceIndex = i

        return maxDistanceIndex

    def reduce_current_buffer(self, forceSaveLastPoint: bool = False, forceSaveZero: bool = False) -> None:
        """Function to reduce the points in the current buffer. This function gets called automatically when the buffer is full,
        but it can also be called manually if a buffer is partially filled and it should be optimised, such as the end of a dataset.

        Args:
            forceSaveLastPoint (bool, optional): Force save the last point in the buffer. Last point should be saved if it is the last point in the dataset`. Defaults to False.
            forceSaveZero (bool, optional): Force save when a point goes to and from a value of zero. Defaults to False.
        """
        # The part of the algorithm which is performing the point reduction.
        # The force reduce parameter needs to be passed as well to force save the last point in the buffer.
        # This is important if a partially full buffer is being reduced. The last point should be saved.

        # Don't run if there are not enough points
        if len(self.buffer.points) <= 2:
            print("ERROR:Trying to reduce less than 3 points, minimum 3 is required")

        if self.runOnLogic:
            # save a copy of the buffer before clearing it
            buffer_copy = copy.copy(self.buffer)

        # Run the reducing function
        self.reduce_buffer(0, self.buffer.get_num_points() - 1)

        if self.forceSaveLastPoint or forceSaveLastPoint:
            self.buffer.points[-1].keep = True

        # Force Save Zero Points
        # if a zero has a nonzero value to the left or right of it, save it
        if self.forceSaveZero or forceSaveZero:
            for i in range(1, len(self.buffer.points) - 1):
                if (self.buffer.points[i].value == 0 and (self.buffer.points[i - 1].value != 0 or self.buffer.points[i + 1].value != 0)):
                    self.buffer.points[i].keep = True

        lastSavedPointIndex = self.log_points()

        # In order to do the "Run-on" curve the last saved point from the previous buffer becomes the first point
        # in the new buffer. The last saved point might not be the last one in the buffer, so the point between the
        # last saved point, and the last point in the buffer will also be saved.
        if self.runOnLogic:
            higherErrorPointIndex = self.get_point_with_highest_error(
                lastSavedPointIndex, buffer_copy.get_num_points() - 1, buffer_copy)

            # Let the algorithm in next buffer run decide if those points should be kept
            buffer_copy.points[lastSavedPointIndex].keep = False
            buffer_copy.points[higherErrorPointIndex].keep = False

            drop_point = False
            if self.bufferSize - lastSavedPointIndex == self.bufferSize:
                point_spread = int(
                    (buffer_copy.points[-1].dateTime - buffer_copy.points[lastSavedPointIndex].dateTime)) % (self.bufferSize - 3) + 1
                point_to_drop_index = lastSavedPointIndex + point_spread

                # check to make sure the highest error or the last saved point never gets dropped
                if point_to_drop_index == higherErrorPointIndex or point_to_drop_index == lastSavedPointIndex:
                    point_to_drop_index += 1

                drop_point = True

            for i in range(lastSavedPointIndex, len(buffer_copy.points)):
                if drop_point and i == point_to_drop_index:
                    continue
                self.buffer.add(buffer_copy.points[i])

    def add_point(self, dateTime: float, value: float, forceReduce: bool = False) -> None:
        """Main function that is used to add points to the curve. When the buffer is full,
        the reduce function is called, which marks which points are to be kept and which to discard
        and then the log points function is called, which will add those points to the bigger buffer


        Args:
            dateTime (float): x value
            value (float): y value
            forceReduce (bool, optional): Flag inidcating if the buffer should be reduced even if it is not full. Defaults to False.
        """
        # First check that the internal state of the curve should not be reset
        if self.setCurveToBeReset:
            # set to False to make sure it does not keep reseting
            self.setCurveToBeReset = False

            # reset the state of the curve
            self.buffer = CurveBuffer(self.bufferSize)
            self.loggedPoints = []

        if self.buffer.get_num_points() == 0:
            firstPoint = BufferedPoint(dateTime, value, keep=False)
            self.buffer.add(firstPoint)
            self.add_logged_point(firstPoint)     # Log First Point
        else:
            self.buffer.add(BufferedPoint(dateTime, value, keep=False))

        if self.buffer.is_full() or forceReduce:
            self.reduce_current_buffer(forceReduce)

    def distance(self, current_point: BufferedPoint, start_point: BufferedPoint, end_point: BufferedPoint) -> float:
        if self.errorType == Distance.VERTICAL:
            d = self.vertical_value(current_point, start_point, end_point)
        elif self.errorType == Distance.PERPENDICULAR:
            d = self.perpendicular_value(current_point, start_point, end_point)
        elif self.errorType == Distance.GPS:
            d = self.gps_distance(current_point, start_point, end_point)
        else:
            print("Invalid distance enum used.")

        return d

    def vertical_value(self, currentPoint: BufferedPoint, firstPoint: BufferedPoint, lastPoint: BufferedPoint) -> float:
        """  Vertical error calculation without C variable size enforced.

        Args:
            currentPoint (BufferedPoint): A point from which the distance should be calculated.
            firstPoint (BufferedPoint): First point in the line
            lastPoint (BufferedPoint): Last point in the line 

        Returns:
            float: The distance
        """
        if lastPoint.dateTime - firstPoint.dateTime != 0:
            dy = (lastPoint.value - firstPoint.value)
            dx = (lastPoint.dateTime - firstPoint.dateTime)
            m = float(dy)/dx
            dxPoint = currentPoint.dateTime - firstPoint.dateTime
            error = abs(currentPoint.value - (firstPoint.value + m * dxPoint))

            if dx < 0:
                print("The data input is not sorted by the x value")

            return error
        else:
            return 0

    # Same logic as the C code, but the variables are defined as float type by Python
    def perpendicular_value(self, currentPoint, firstPoint, lastPoint):
        """  Perpendicular error calculation without C variable size enforced.

        Args:
            currentPoint (BufferedPoint): A point from which the distance should be calculated.
            firstPoint (BufferedPoint): First point in the line
            lastPoint (BufferedPoint): Last point in the line 

        Returns:
            float: The distance
        """
        p1x = firstPoint.dateTime
        p2x = currentPoint.dateTime
        p3x = lastPoint.dateTime

        p1y = firstPoint.value
        p2y = currentPoint.value
        p3y = lastPoint.value

        c_timeRatio = self.timeRatio
        c_dataRatio = self.dataRatio

        lEnd = (p3x - p1x) * c_timeRatio
        lPoint = (p2x - p1x) * c_timeRatio

        lDataStart = p1y / c_dataRatio
        lDataPoint = p2y / c_dataRatio
        lDataEnd = p3y / c_dataRatio

        lArea = (lEnd * (lDataPoint - lDataStart)) + \
            (lPoint * (lDataStart - lDataEnd))
        if (lArea == 0):
            return 0

        dBase = math.sqrt(
            (lEnd * lEnd) + (((lDataStart - lDataEnd) * (lDataStart - lDataEnd))))
        if dBase > 0.0:
            height = abs(lArea) / dBase
        else:
            height = 0

        return float(height)

    def gps_distance(self, currentPoint: BufferedPoint, firstPoint: BufferedPoint, lastPoint: BufferedPoint) -> float:
        """  GPS error calculation without C variable size enforced.

        Args:
            currentPoint (BufferedPoint): A point from which the distance should be calculated.
            firstPoint (BufferedPoint): First point in the line
            lastPoint (BufferedPoint): Last point in the line 

        Returns:
            float: The distance
        """
        start_lat = firstPoint.dateTime
        start_long = firstPoint.value
        current_lat = currentPoint.dateTime
        current_long = currentPoint.value
        end_lat = lastPoint.dateTime
        end_long = lastPoint.value

        area = abs((start_long * (end_lat - current_lat) + current_long * (start_lat - end_lat) +
                    end_long * (current_lat - start_lat)) * math.cos(((start_lat + end_lat)/2))*(math.pi/180))
        if area == 0:
            return 0

        base = self.estimate_lat_long_distance(
            start_lat, start_long, end_lat, end_long)

        if base > 0:
            distance = area/base
        else:
            distance = 0

        distance *= 1e6  # multiplied to be in meters

        return distance

    def estimate_lat_long_distance(self, start_lat: float, start_long: float, end_lat: float, end_long: float) -> float:
        # Estimate the distance between points
        # https://www.movable-type.co.uk/scripts/latlong.html
        start_lat_rad = start_lat*(math.pi/180)
        end_lat_rad = end_lat*(math.pi/180)
        start_long_rad = start_long*(math.pi/180)
        end_long_rad = end_long*(math.pi/180)

        x = (end_long_rad - start_long_rad) * \
            math.cos((start_lat_rad + end_lat_rad)/2)
        y = end_lat_rad - start_lat_rad
        return math.sqrt(x**2 + y**2)

    def reduce_buffer(self, firstIndex: int, lastIndex: int):
        """A helper function responsible to optimise the buffer.

        Args:
            firstIndex (int): Index of the first point in the buffer from which the optimization should start
            lastIndex (int): Index of the last point in the buffer up to which the optimization should go to
        """
        # Skip the computation if the start and end don't have enough points in between
        if lastIndex - firstIndex <= 1:
            return

        largest_distance = 0.0
        index_of_largest_distance = firstIndex

        # Loop through all the points between the first and last finding the one with highest error
        for i in range(index_of_largest_distance, lastIndex + 1):
            dist = self.distance(
                self.buffer.points[i], self.buffer.points[firstIndex], self.buffer.points[lastIndex])

            if dist > largest_distance:
                index_of_largest_distance = i
                largest_distance = dist

        # if there are any points between the start and end, do a recursive call
        # so the points between will be analyzed
        if largest_distance > self.allowedError:
            self.buffer.points[index_of_largest_distance].keep = True
            self.reduce_buffer(index_of_largest_distance, lastIndex)
            self.reduce_buffer(firstIndex, index_of_largest_distance)
        else:
            # Otherwise just mark all the points between start and end as points which should not be kept
            for i in range(firstIndex + 1, lastIndex):
                self.buffer.points[i].keep = False

    # Return the buffer that holds all the points which are kept
    # if the resetCurve argument is true, the internal state of the curve will be reset
    #
    # Note: The reset of the state will occur when the next point is added to the buffer.
    # otherwise a copy of the buffer would be required which would increase memory consumption
    def get_reduced_points(self, resetCurve: bool = False) -> list:
        """Get a copy of the points which have been reduced and optimised by the curve

        Args:
            resetCurve (bool, optional): Flag indicating if the curve should be reset (All the optimised points discarded). If enabled, this will return None. Defaults to False.

        Returns:
            list: List of points optimised
        """
        # This statement is checking if the curve is already set to be cleared.
        # If the curve was already set to be cleared, just return None. This case is important if
        # GetReducedPoints(True) gets called, and then GetReducedPoints(*) gets called again
        # the first time it should return the points as expected, the second time, it should
        # return None, since the curve is meant to be reset. (memory is only cleared on next point add)
        # If GetReducedPoints(False) is called first, and then GetReducedPoints(*) it will return the buffer both times.
        if self.setCurveToBeReset:
            return None

        self.setCurveToBeReset = resetCurve
        return self.loggedPoints
