import threading

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def moving_average(data, window_size):
    """Return a valid convolution of data with a uniform moving-average window."""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="valid")


@jit(nopython=True, cache=True)
def rowwise_max(arr):
    """Return the maximum of each column in a two-dimensional array."""
    n = arr.shape[1]
    out = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        out[i] = np.max(arr[:, i])
    return out


@jit(nopython=True, cache=True)
def rowwise_mean(arr):
    """Return the mean of each column in a two-dimensional array."""
    n = arr.shape[1]
    out = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        out[i] = np.mean(arr[:, i])
    return out


class SpectrumDataStorage:
    """Store spectrum samples in a fixed-size, thread-safe circular buffer."""

    def __init__(self, buffer_size=1):
        """Initialize storage for up to buffer_size spectrum samples."""
        self.buffer_size = buffer_size
        self._smooth_window_size = 4

        self._x = np.arange(2399.0, 2485.5, 0.5)
        self._y = np.zeros(buffer_size, dtype=int)
        self._z = np.full((buffer_size, len(self._x)), -120, dtype=float)
        self._z_smooth = np.zeros(
            (buffer_size, len(self._x) - self._smooth_window_size + 1), dtype=float
        )

        self._current_size = 0
        self._next_idx = 0

        self.lock = threading.RLock()

    @property
    def x(self):
        """Return a copy of the frequency coordinates."""
        return self._x.copy()

    @property
    def y(self):
        """Return timestamps for the samples currently in the buffer."""
        with self.lock:
            return self._y[: self._current_size].copy()

    @property
    def z(self):
        """Return spectrum samples currently in the buffer."""
        with self.lock:
            return self._z[: self._current_size].copy()

    @property
    def shape(self):
        """Return the frequency and current buffer dimensions."""
        with self.lock:
            return (
                len(self._x),
                self._y[: self._current_size].shape[0],
                self._z[: self._current_size].shape[0],
                self._z.shape[1],
            )

    @property
    def smooth_window_size(self):
        """Return the moving-average window size."""
        with self.lock:
            return self._smooth_window_size

    def set_smooth_window_size(self, value, recompute=False):
        """Change the moving-average window size.

        Args:
            value: New window size, at least 1.
            recompute: Whether to recompute the smoothed data of stored samples.
                Without it, stored smoothed samples are cleared.

        Raises:
            ValueError: If value is lower than 1 or larger than the number of
                frequency coordinates.
        """
        if value < 1 or value > len(self._x):
            raise ValueError("smooth window_size must be between 1 and len(x)")
        with self.lock:
            self._smooth_window_size = value
            self._z_smooth = np.zeros(
                (self.buffer_size, len(self._x) - value + 1), dtype=float
            )
            if recompute and self._current_size:
                for i in range(self._current_size):
                    self._z_smooth[i] = moving_average(self._z[i], value)

    def _last_rows(self, array, count):
        """Return the last count stored rows of array, oldest first.

        The result may be a view of the buffer: use it while holding the lock.
        """
        count = min(count, self._current_size)
        end = self._next_idx
        if count <= end:
            return array[end - count : end]
        return np.concatenate((array[end - count :], array[:end]))

    def _count(self, number, name):
        """Validate a requested row count and default it to the stored size."""
        if number is None:
            return self._current_size
        if number > self.buffer_size:
            raise ValueError(f"{name} cannot be greater than buffer_size")
        return number

    def append(self, y, z):
        """Add a timestamp and spectrum sample to the circular buffer.

        Args:
            y: Timestamp associated with the sample.
            z: Spectrum values, with one value per frequency coordinate.

        Raises:
            ValueError: If z does not match the frequency-coordinate length.
        """
        if len(z) != len(self._x):
            raise ValueError("Length of z must be equal to length of x")

        with self.lock:
            idx = self._next_idx
            self._y[idx] = y
            self._z[idx] = z
            self._z_smooth[idx] = moving_average(self._z[idx], self._smooth_window_size)

            if self._current_size < self.buffer_size:
                self._current_size += 1
            self._next_idx = (idx + 1) % self.buffer_size

    def get(self):
        """Return the newest raw spectrum sample, or None if the buffer is empty."""
        with self.lock:
            if self._current_size == 0:
                return None
            return self._z[self._next_idx - 1].copy()

    def get_smooth(self):
        """Return smoothed frequency coordinates and the newest smoothed sample.

        If the buffer is empty, the smoothed sample is None.
        """
        with self.lock:
            x = self._x[self._smooth_window_size - 1 :]
            if self._current_size == 0:
                return x, None
            return x, self._z_smooth[self._next_idx - 1].copy()

    def get_max(self, number=None):
        """Return per-frequency maxima across the most recent samples.

        Args:
            number: Number of recent samples to include, or None for all stored
                samples.

        Returns:
            The per-frequency maxima, or None if the buffer is empty.

        Raises:
            ValueError: If number exceeds the buffer capacity.
        """
        with self.lock:
            if self._current_size == 0:
                return None
            number = self._count(number, "number")
            return rowwise_max(self._last_rows(self._z, number))

    def get_mean(self, number=None):
        """Return per-frequency means across the most recent samples.

        Args:
            number: Number of recent samples to include, or None for all stored
                samples.

        Returns:
            The per-frequency means, or None if the buffer is empty.

        Raises:
            ValueError: If number exceeds the buffer capacity.
        """
        with self.lock:
            if self._current_size == 0:
                return None
            number = self._count(number, "number")
            return rowwise_mean(self._last_rows(self._z, number))

    def get_data(self, nb_value=None, pad_value=0):
        """Return raw samples in chronological order, optionally padding rows.

        Args:
            nb_value: Number of rows to return, or None for the number stored.
            pad_value: Value used to fill rows preceding the stored samples.

        Raises:
            ValueError: If nb_value exceeds the buffer capacity.
        """
        with self.lock:
            if self._current_size == 0:
                return np.zeros((nb_value if nb_value else 0, len(self._x)))
            nb_value = self._count(nb_value, "nb_value")
            data = self._last_rows(self._z, nb_value).copy()
        if nb_value > data.shape[0]:
            padding = ((nb_value - data.shape[0], 0), (0, 0))
            data = np.pad(data, padding, mode="constant", constant_values=pad_value)
        return data

    def histogram(self, nb_value=None, bins=10, data_range=None, clip=False):
        """Count, for each frequency, how many samples fall in each value bin.

        Args:
            nb_value: Number of recent samples to include, or None for all stored
                samples.
            bins: Bin count or bin edges, as accepted by numpy.histogram_bin_edges.
            data_range: Optional value range used when bins is a count.
            clip: Whether to count values outside the bin edges in the first or
                last bin instead of ignoring them.

        Returns:
            A pair containing the bin-by-frequency counts and the bin edges. The
            bin edges are None when the buffer is empty.

        Raises:
            ValueError: If nb_value exceeds the buffer capacity.
        """
        with self.lock:
            if self._current_size == 0:
                nb_bins = bins if np.isscalar(bins) else len(bins) - 1
                return np.zeros((nb_bins, len(self._x))), None
            nb_value = self._count(nb_value, "nb_value")
            data = self._last_rows(self._z, nb_value).copy()

        edges = np.histogram_bin_edges(data, bins=bins, range=data_range)
        nb_bins = len(edges) - 1
        n_freq = data.shape[1]
        if clip:
            data = np.clip(data, edges[0], edges[-1])
        # Like numpy.histogram, the last bin includes its right edge.
        idx = np.searchsorted(edges, data, side="right") - 1
        idx[data == edges[-1]] = nb_bins - 1
        valid = (idx >= 0) & (idx < nb_bins)
        flat = idx + nb_bins * np.arange(n_freq)
        counts = np.bincount(flat[valid], minlength=nb_bins * n_freq)
        return counts.reshape(n_freq, nb_bins).T, edges

    def _padded_y(self, nb_value):
        """Return timestamps for nb_value rows, extrapolating missing leading ones."""
        with self.lock:
            y_data = self._y[: self._current_size].copy()
        if nb_value > y_data.shape[0]:
            padding_size = nb_value - y_data.shape[0]
            if y_data.shape[0] > 0:
                interval = np.median(np.diff(y_data)) if y_data.shape[0] > 1 else 1
                pad_start = y_data[0] - interval * padding_size
                padding = np.arange(pad_start, y_data[0], interval, dtype=int)
            else:
                padding = np.zeros(padding_size, dtype=int)
            y_data = np.pad(y_data, (padding_size, 0), mode="constant")
            y_data[:padding_size] = padding
        return y_data

    def get_3D(self, nb_value=None, pad_value=0):
        """Return frequency coordinates, padded timestamps, and raw samples.

        Args:
            nb_value: Number of rows to return, or None for the number stored.
            pad_value: Value used to fill leading rows when padding is needed.

        Raises:
            ValueError: If nb_value exceeds the buffer capacity.
        """
        z_data = self.get_data(nb_value, pad_value)
        return self._x, self._padded_y(z_data.shape[0]), z_data
