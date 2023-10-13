# -*- coding: utf-8 -*-
from __future__ import annotations
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from .util.signal import Signal
from .auto_background import AutoBackground, SmoothBrucknerBackground


class Pattern(object):
    """
    A Pattern is a set of x and y values.
    It can be loaded from a file or created from scratch and can be modified by
    different methods.
    It builds the basis for all calculations in glassure.

    :param x: x values of the pattern
    :param y: y values of the pattern
    :param name: name of the pattern
    """

    def __init__(self, x: np.ndarray = None, y: np.ndarray = None, name: str = ''):
        """
        Creates a new Pattern object, x and y should have the same shape.
        """
        if x is None:
            self._original_x = np.linspace(0.1, 15, 100)
        else:
            self._original_x = x
        if y is None:
            self._original_y = np.log(self._original_x ** 2) - (self._original_x * 0.2) ** 2
        else:
            self._original_y = y

        self.name = name
        self._offset = 0.0
        self._scaling = 1.0
        self._smoothing = 0.0
        self._background_pattern = None

        self._auto_bkg: AutoBackground = None

        self._pattern_x = self._original_x
        self._pattern_y = self._original_y

        self._auto_background_before_subtraction_pattern = None
        self._auto_background_pattern = None

        self.changed = Signal()

    def load(self, filename: str, skiprows: int = 0):
        """
        Loads a pattern from a file. The file can be either a .xy or a .chi file. The .chi file will be loaded with
        skiprows=4 by default.

        :param filename: path to the file
        :param skiprows: number of rows to skip when loading the data (header)
        """
        try:
            if filename.endswith('.chi'):
                skiprows = 4
            data = np.loadtxt(filename, skiprows=skiprows)
            self._original_x = data.T[0]
            self._original_y = data.T[1]
            self.name = os.path.basename(filename).split('.')[:-1][0]
            self.recalculate_pattern()

        except ValueError:
            raise ValueError('Wrong data format for pattern file! - ' + filename)

    @staticmethod
    def from_file(filename: str, skip_rows: int = 0) -> Pattern | '-1':
        """
        Loads a pattern from a file. The file can be either a .xy or a .chi file. The .chi file will be loaded with
        skiprows=4 by default.

        :param filename: path to the file
        :param skip_rows: number of rows to skip when loading the data (header)
        """
        try:
            if filename.endswith('.chi'):
                skip_rows = 4
            data = np.loadtxt(filename, skiprows=skip_rows)
            x = data.T[0]
            y = data.T[1]
            name = os.path.basename(filename).split('.')[:-1][0]
            return Pattern(x, y, name)

        except ValueError:
            raise ValueError('Wrong data format for pattern file! - ' + filename)

    def save(self, filename, header='', subtract_background=False, unit='2th_deg'):
        """
        Saves the x, y data to file. Supporting several file formats: .chi, .xy, .fxye
        :param filename: where to save the data
        :param header: header for file
        :param subtract_background: whether to save subtracted data
        :param unit: x-unit used for the standard chi header (unused for other formats)
        """
        x, y = self.data
        if not subtract_background and self.background_pattern is not None:
            y += self.background_pattern.y

        num_points = len(x)

        file_handle = open(filename, 'w')

        if filename.endswith('.chi'):
            if header is None or header == '':
                file_handle.write(filename + '\n')
                file_handle.write(unit + '\n\n')
                file_handle.write("       {0}\n".format(num_points))
            else:
                file_handle.write(header)
            for ind in range(num_points):
                file_handle.write(' {0:.7E}  {1:.7E}\n'.format(x[ind], y[ind]))
        elif filename.endswith('.fxye'):
            factor = 100
            if 'CONQ' in header:
                factor = 1
            header = header.replace('NUM_POINTS', '{0:.6g}'.format(num_points))
            header = header.replace('MIN_X_VAL', '{0:.6g}'.format(factor * x[0]))
            header = header.replace('STEP_X_VAL', '{0:.6g}'.format(factor * (x[1] - x[0])))

            file_handle.write(header)
            file_handle.write('\n')
            for ind in range(num_points):
                file_handle.write('\t{0:.6g}\t{1:.6g}\t{2:.6g}\n'.format(factor * x[ind], y[ind], np.sqrt(abs(y[ind]))))
        else:

            data = np.dstack((x, y))
            np.savetxt(file_handle, data[0], header=header)
        file_handle.close()

    @property
    def background_pattern(self) -> Pattern:
        """
        Returns the background pattern of the current pattern.
        :return: background Pattern
        """
        return self._background_pattern

    @background_pattern.setter
    def background_pattern(self, pattern: Pattern | None):
        if self._background_pattern is not None:
            self._background_pattern.changed.disconnect(self.recalculate_pattern)

        self._background_pattern = pattern
        if self._background_pattern is not None:
            self._background_pattern.changed.connect(self.recalculate_pattern)
        self.recalculate_pattern()

    def rebin(self, bin_size: float) -> Pattern:
        """
        Returns a new pattern, which is a rebinned version of the current one.

        :param bin_size: Size of the bins
        :return: rebinned Pattern
        """
        x, y = self.data
        x_min = np.round(np.min(x) / bin_size) * bin_size
        x_max = np.round(np.max(x) / bin_size) * bin_size
        new_x = np.arange(x_min, x_max + 0.1 * bin_size, bin_size)

        bins = np.hstack((x_min - bin_size * 0.5, new_x + bin_size * 0.5))
        new_y = (np.histogram(x, bins, weights=y)
                 [0] / np.histogram(x, bins)[0])

        return Pattern(new_x, new_y)

    @property
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the data of the pattern. If a background pattern is set, the background will be subtracted from the
        pattern. If smoothing is set, the pattern will be smoothed.

        :return: Tuple of x and y values
        """
        return self._pattern_x, self._pattern_y

    def recalculate_pattern(self):
        """
        Returns the data of the pattern. If a background pattern is set, the background will be subtracted from the
        pattern. If smoothing is set, the pattern will be smoothed.

        :return: Tuple of x and y values
        """
        if self._background_pattern is not None:
            # create background function
            x_bkg, y_bkg = self._background_pattern.data

            if not np.array_equal(x_bkg, self._original_x):
                # the background will be interpolated
                f_bkg = interp1d(x_bkg, y_bkg, kind='linear')

                # find overlapping x and y values:
                ind = np.where((self._original_x <= np.max(x_bkg)) &
                               (self._original_x >= np.min(x_bkg)))
                x = self._original_x[ind]
                y = self._original_y[ind]

                if len(x) == 0:
                    # if there is no overlapping between background and pattern, raise an error
                    raise BkgNotInRangeError(self.name)

                y = y * self._scaling + self.offset - f_bkg(x)
            else:
                # if pattern and bkg have the same x basis we just delete y-y_bkg
                x, y = self._original_x, self._original_y * self._scaling + self.offset - y_bkg
        else:
            x, y = self.original_data
            y = y * self.scaling + self.offset

        if self._auto_bkg is not None:
            self._auto_background_before_subtraction_pattern = Pattern(x, y)
            x, y = self._auto_background_before_subtraction_pattern.limit(*self._auto_bkg_roi).data
            y_bkg = self._auto_bkg.extract_background(Pattern(x, y))
            self._auto_background_pattern = Pattern(x, y_bkg, name='auto_bkg_' + self.name)
            y -= y_bkg

        if self.smoothing > 0:
            y = gaussian_filter1d(y, self.smoothing)

        self._pattern_x = x
        self._pattern_y = y
        self.changed.emit()

    @data.setter
    def data(self, data: tuple[np.ndarray, np.ndarray]):
        """
        Sets the data of the pattern. Also resets the scaling and offset to 1 and 0 respectively.

        :param data: tuple of x and y values
        """
        (x, y) = data
        self._original_x = x
        self._original_y = y
        self.scaling = 1.0
        self.offset = 0

    @property
    def original_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the original data of the pattern without any background subtraction or smoothing.

        :return: tuple of x and y values
        """
        return self._original_x, self._original_y

    @property
    def x(self) -> np.ndarray:
        """ Returns the x values of the pattern """
        return self._pattern_x

    @x.setter
    def x(self, new_value: np.ndarray):
        """ Sets the x values of the pattern """
        self._original_x = new_value
        self.recalculate_pattern()

    @property
    def y(self) -> np.ndarray:
        """ Returns the y values of the pattern """
        return self._pattern_y

    @y.setter
    def y(self, new_y: np.ndarray):
        """ Sets the y values of the pattern """
        self._original_y = new_y
        self.recalculate_pattern()

    @property
    def scaling(self) -> float:
        """ Returns the scaling of the pattern """
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        """ 
        Sets the scaling of the pattern, if below 0, it will be set to 0 
        instead.
        """
        if value < 0:
            self._scaling = 0.0
        else:
            self._scaling = value
        self.recalculate_pattern()

    @property
    def offset(self) -> float:
        """ Returns the offset of the pattern """
        return self._offset

    @offset.setter
    def offset(self, value):
        """ Sets the offset of the pattern """
        self._offset = value
        self.recalculate_pattern()

    @property
    def smoothing(self) -> float:
        """ Returns the smoothing of the pattern """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        """ Sets the smoothing of the pattern """
        self._smoothing = value
        self.recalculate_pattern()

    def set_auto_background_subtraction(self, parameters: list[float], roi: list[float] = None, recalc_pattern=True):
        """
        Sets an automatic background subtraction to the pattern. The background will be subtracted from the pattern
        when calling the data property.

        :param parameters: list of parameters for the automatic background subtraction. The first parameter is the
        smoothing amount, the second, the number of iterations for the bruckner smoothing and the third the order of the
        chebyshev polynomial.
        :param roi: region of interest to be used for  the background subtraction. If None, the whole pattern will be
        used.
        """
        if roi is None:
            min_step = self.x[1] - self.x[0]
            max_step = self.x[-1] - self.x[-2]
            roi = [np.min(self.x) - min_step, np.max(self.x) + max_step]
        self._auto_bkg = SmoothBrucknerBackground(*parameters)
        self._auto_bkg_roi = roi
        if recalc_pattern:
            self.recalculate_pattern()

    def unset_auto_background_subtraction(self):
        self._auto_bkg = None
        self.recalculate_pattern()

    @property
    def auto_background_pattern(self) -> Pattern:
        """
        Returns the auto background pattern
        :return: background Pattern
        """
        return self._auto_background_pattern

    @property
    def auto_background_before_subtraction_pattern(self) -> Pattern:
        """
        Returns the pattern before the auto background subtraction
        :return: background Pattern
        """
        return self._auto_background_before_subtraction_pattern

    def limit(self, x_min: float, x_max: float) -> Pattern:
        """
        Limits the pattern to a specific x-range. Does not modify inplace but returns a new limited Pattern

        :param x_min: lower limit of the x-range
        :param x_max: upper limit of the x-range
        :return: limited Pattern
        """
        x, y = self.data
        return Pattern(x[np.where((x_min < x) & (x < x_max))],
                       y[np.where((x_min < x) & (x < x_max))])

    def extend_to(self, x_value: float, y_value: float) -> Pattern:
        """
        Extends the current pattern to a specific x_value by filling it with the y_value. Does not modify inplace but
        returns a new filled Pattern

        :param x_value: Point to which extending the pattern should be smaller than the lowest x-value in the pattern or
        vice versa
        :param y_value: number to fill the pattern with
        :return: extended Pattern
        """
        x_step = np.mean(np.diff(self.x))
        x_min = np.min(self.x)
        x_max = np.max(self.x)
        if x_value < x_min:
            x_fill = np.arange(x_min - x_step, x_value -
                               x_step * 0.5, -x_step)[::-1]
            y_fill = np.zeros(x_fill.shape)
            y_fill.fill(y_value)

            new_x = np.concatenate((x_fill, self.x))
            new_y = np.concatenate((y_fill, self.y))
        elif x_value > x_max:
            x_fill = np.arange(x_max + x_step, x_value + x_step * 0.5, x_step)
            y_fill = np.zeros(x_fill.shape)
            y_fill.fill(y_value)

            new_x = np.concatenate((self.x, x_fill))
            new_y = np.concatenate((self.y, y_fill))
        else:
            return self

        return Pattern(new_x, new_y)

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the pattern which can be used to save the pattern to a json file.

        :return: dictionary representation of the pattern
        """
        return {
            'name': self.name,
            'x': self._original_x.tolist(),
            'y': self._original_y.tolist(),
            'scaling': self.scaling,
            'offset': self.offset,
            'smoothing': self.smoothing,
            'bkg_pattern': self._background_pattern.to_dict() if self._background_pattern is
                                                                 not None else None
        }

    @staticmethod
    def from_dict(json_dict: dict) -> Pattern:
        """
        Creates a new Pattern from a dictionary representation of a Pattern.

        :param json_dict: dictionary representation of a Pattern
        :return: new Pattern
        """
        pattern = Pattern(np.array(json_dict['x']),
                          np.array(json_dict['y']),
                          json_dict['name'])

        pattern.scaling = json_dict['scaling']
        pattern.offset = json_dict['offset']

        if json_dict['bkg_pattern'] is not None:
            bkg_pattern = Pattern.from_dict(json_dict['bkg_pattern'])
        else:
            bkg_pattern = None
        pattern.background_pattern = bkg_pattern

        pattern.smoothing = json_dict['smoothing']
        pattern.recalculate_pattern()

        return pattern

    ###########################################################
    # Operators:

    def __sub__(self, other: Pattern) -> Pattern:
        """
        Subtracts the other pattern from the current one. If the other pattern
        has a different shape, the subtraction will be done on the overlapping
        x-values and the background will be interpolated. If there is no
        overlapping between the two patterns, a BkgNotInRangeError will be
        raised.

        :param other: Pattern to be subtracted
        :return: new Pattern
        """
        orig_x, orig_y = self.data
        other_x, other_y = other.data

        if orig_x.shape != other_x.shape:
            # the background will be interpolated
            other_fcn = interp1d(other_x, other_y, kind='cubic')

            # find overlapping x and y values:
            ind = np.where((orig_x <= np.max(other_x)) & (orig_x >= np.min(other_x)))
            x = orig_x[ind]
            y = orig_y[ind]

            if len(x) == 0:
                # if there is no overlapping between background and pattern, raise an error
                raise BkgNotInRangeError(self.name)
            return Pattern(x, y - other_fcn(x))
        else:
            return Pattern(orig_x, orig_y - other_y)

    def __add__(self, other: Pattern) -> Pattern:
        """
        Adds the other pattern to the current one. If the other pattern
        has a different shape, the addition will be done on the overlapping
        x-values and the y-values of the other pattern will be interpolated. 
        If there is no overlapping between the two patterns, a BkgNotInRangeror 
        will be raised.

        :param other: Pattern to be added
        :return: new Pattern
        """
        orig_x, orig_y = self.data
        other_x, other_y = other.data

        if orig_x.shape != other_x.shape:
            # the background will be interpolated
            other_fcn = interp1d(other_x, other_y, kind='linear')

            # find overlapping x and y values:
            ind = np.where((orig_x <= np.max(other_x)) & (orig_x >= np.min(other_x)))
            x = orig_x[ind]
            y = orig_y[ind]

            if len(x) == 0:
                # if there is no overlapping between background and pattern, raise an error
                raise BkgNotInRangeError(self.name)
            return Pattern(x, y + other_fcn(x))
        else:
            return Pattern(orig_x, orig_y + other_y)

    def __rmul__(self, other: float) -> Pattern:
        """
        Multiplies the pattern with a scalar.

        :param other: scalar to multiply with
        :return: new Pattern
        """
        orig_x, orig_y = self.data
        return Pattern(np.copy(orig_x), np.copy(orig_y) * other)

    def __eq__(self, other: Pattern) -> bool:
        """
        Checks if two patterns are equal. Two patterns are equal if their data
        is equal.

        :param other: Pattern to compare with
        :return: True if equal, False otherwise
        """
        if not isinstance(other, Pattern):
            return False
        if np.array_equal(self.data, other.data):
            return True
        return False

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f'Pattern \'{self.name}\' with {len(self)} points'


class BkgNotInRangeError(Exception):
    def __init__(self, pattern_name: str):
        self.pattern_name = pattern_name

    def __str__(self):
        return "The background range does not overlap with the Pattern range for " + self.pattern_name
