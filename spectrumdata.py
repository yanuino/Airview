import numpy as np
from numba import jit
import threading

@jit(nopython=True)
def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

@jit(nopython=True)
def rowwise_max(arr):
    n = arr.shape[1]
    out = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        out[i] = np.max(arr[:, i])
    return out

@jit(nopython=True)
def rowwise_mean(arr):
    n = arr.shape[1]
    out = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        out[i] = np.mean(arr[:, i])
    return out

class SpectrumDataStorage:
    def __init__(self, buffer_size=1):
        self.buffer_size = buffer_size
        self._max_window_size = 5
        self._smooth_window_size = 4

        self._x = np.arange(2399.0, 2485.5, 0.5)
        self._y = np.zeros(buffer_size, dtype=int)
        self._z = np.full((buffer_size, len(self._x)), -120, dtype=float)
        self._z_max = np.zeros((buffer_size, len(self._x)), dtype=float)
        self._z_smooth = np.zeros((buffer_size, len(self._x[self._smooth_window_size -1:])), dtype=float)

        self._current_size = 0
        self._next_idx = 0

        self.lock = threading.RLock()

    @property
    def x(self):
        return self._x.copy()
        
    @property
    def y(self):
        with self.lock:
            return self._y[:self._current_size].copy()
        
    @property
    def z(self):
        with self.lock:
            return self._z[:self._current_size].copy()

    @property
    def shape(self):
        with self.lock:
            return len(self._x), self._y[:self._current_size].shape[0], self._z[:self._current_size].shape[0], self._z.shape[1]
    
    @property
    def max_window_size(self):
        with self.lock:
            return self._max_window_size
    
    @max_window_size.setter
    def max_window_size(self, value, recompute=False):
        if value < 1:
            raise ValueError("max window_size must be at least 1")
        with self.lock:
            self._max_window_size = value
        if recompute:
            self.__recompute_z_max()       

    @property
    def smooth_window_size(self):
        with self.lock:
            return self._smooth_window_size
    
    @smooth_window_size.setter
    def smooth_window_size(self, value, recompute=False):
        if value < 1:
            raise ValueError("smooth window_size must be at least 1")
        with self.lock:
            self._smooth_window_size = value
        if recompute:
            self.__recompute_z_smooth()       

    def __moving_average(self, data):
        return moving_average(data, self._smooth_window_size)
    
    def __recompute_z_max(self):
        with self.lock:
            z_max_list = []
            for i in range(self._current_size):
                if i < self._max_window_size:
                    max_z = np.max(self._z[:i + 1], axis=0)
                else:
                    max_z = np.max(self._z[i - self._max_window_size + 1:i + 1], axis=0)
                z_max_list.append(max_z)
            self._z_max[:self._current_size] = np.vstack(z_max_list)

    def __recompute_z_smooth(self):
        with self.lock:
            smoothed_data = []
            for i in range(self._current_size):
                smoothed_data.append(self.__moving_average(self._z[i]))
            self._z_smooth[:self._current_size] = np.vstack(smoothed_data)

    def append(self, y, z):
        if len(z) != len(self._x):
            raise ValueError("Length of z must be equal to length of x")

        with self.lock:
            idx = self._next_idx
            self._y[idx] = y
            self._z[idx] = z
            self._z_smooth[idx] = self.__moving_average(z)

            # Compute z_max for this entry
            if self._current_size < self._max_window_size:
                max_z = np.max(self._z[:self._current_size + 1], axis=0)
            else:
                start_idx = (idx - self._max_window_size + 1) % self.buffer_size
                if self._current_size < self.buffer_size:
                    max_z = np.max(self._z[:self._current_size + 1], axis=0)
                else:
                    if start_idx <= idx:
                        max_z = np.max(self._z[start_idx:idx + 1], axis=0)
                    else:
                        max_z = np.max(np.vstack((self._z[start_idx:], self._z[:idx + 1])), axis=0)
            self._z_max[idx] = max_z

            if self._current_size < self.buffer_size:
                self._current_size += 1
            self._next_idx = (self._next_idx + 1) % self.buffer_size

    def get(self):
        with self.lock:
            if self._current_size == 0:
                return None
            idx = (self._next_idx - 1) % self.buffer_size
            return self._z[idx]
        
    def get_smooth(self):
        with self.lock:
            if self._current_size == 0:
                return self._x[self._smooth_window_size -1:], None
            idx = (self._next_idx - 1) % self.buffer_size
            return self._x[self._smooth_window_size -1:], self._z_smooth[idx]
        
    def get_max(self, number=None):
        with self.lock:
            if self._current_size == 0:
                return None
            if number is None:
                number = self._current_size
            if number > self.buffer_size:
                raise ValueError("number cannot be greater than buffer_size")
            idxs = [(self._next_idx - i - 1) % self.buffer_size for i in range(number)]
            data = self._z[idxs]
        return rowwise_max(data)

    def get_mean(self, number=None):
        with self.lock:
            if self._current_size == 0:
                return None
            if number is None:
                number = self._current_size
            if number > self.buffer_size:
                raise ValueError("number cannot be greater than buffer_size")
            idxs = [(self._next_idx - i - 1) % self.buffer_size for i in range(number)]
            data = self._z[idxs]
        return rowwise_mean(data)

    def get_data(self, nb_value=None, pad_value=0):
        with self.lock:
            if self._current_size == 0:
                return np.zeros((nb_value if nb_value else 0, len(self._x)))
            if nb_value is None:
                nb_value = self._current_size
            if nb_value > self.buffer_size:
                raise ValueError("nb_value cannot be greater than buffer_size")
            idxs = [(self._next_idx - i - 1) % self.buffer_size for i in range(nb_value)]
            data = self._z[idxs][::-1]
        if nb_value > data.shape[0]:
            padding = ((nb_value - data.shape[0], 0), (0, 0))
            data = np.pad(data, padding, mode='constant', constant_values=pad_value)
        return data

    def histogram(self, nb_value=None, bins=10, data_range=None):
        with self.lock:
            hist = []
            bin_edges = None
            if self._current_size == 0:
                return np.zeros((bins, len(self._x))), None
            if nb_value is None:
                nb_value = self._current_size
            if nb_value > self.buffer_size:
                raise ValueError("nb_value cannot be greater than buffer_size")
            idxs = [(self._next_idx - i - 1) % self.buffer_size for i in range(nb_value)]
            data = self._z[idxs][::-1]
        for i in range(data.shape[1]):
            h, edges = np.histogram(data[:, i], bins=bins, range=data_range)
            hist.append(h)
            if bin_edges is None:
                bin_edges = edges
        return np.fliplr(np.rot90(np.array(hist), axes=(1,0))), bin_edges

    def __get_maxdata(self, nb_value=None, pad_value=0):
        with self.lock:
            if self._current_size == 0:
                return np.zeros((nb_value if nb_value else 0, len(self._x)))
            if nb_value is None:
                nb_value = self._current_size
            if nb_value > self.buffer_size:
                raise ValueError("nb_value cannot be greater than buffer_size")
            idxs = [(self._next_idx - i - 1) % self.buffer_size for i in range(nb_value)]
            maxdata = self._z_max[idxs][::-1]
        if nb_value > maxdata.shape[0]:
            padding = ((nb_value - maxdata.shape[0], 0), (0, 0))
            maxdata = np.pad(maxdata, padding, mode='constant', constant_values=pad_value)
        return maxdata

    def get_3D(self, nb_value=None, pad_value=0):
        z_data = self.get_data(nb_value, pad_value)
        if nb_value is None:
            nb_value = z_data.shape[0]
        if nb_value > self.buffer_size:
            raise ValueError("nb_value cannot be greater than buffer_size")
        with self.lock:
            y_data = self._y[:self._current_size]
        if nb_value > y_data.shape[0]:
            padding_size = nb_value - y_data.shape[0]
            if y_data.shape[0] > 1:
                interval = np.median(np.diff(y_data))  # Calculate median interval
            else:
                interval = 1  # Default interval if not enough data points
            pad_start = y_data[0] - interval * padding_size if y_data.shape[0] > 0 else 0
            padding = np.arange(pad_start, y_data[0], interval, dtype=int) if y_data.shape[0] > 0 else np.zeros(padding_size, dtype=int)
            y_data = np.pad(y_data, (padding_size, 0), mode='constant', constant_values=pad_value)
            y_data[:padding_size] = padding
        return self._x, y_data, z_data
        
    def get_max3D(self, nb_value=None, pad_value=0):
        z_max = self.__get_maxdata(nb_value, pad_value)
        if nb_value is None:
            nb_value = z_max.shape[0]
        if nb_value > self.buffer_size:
            raise ValueError("nb_value cannot be greater than buffer_size")
        with self.lock:
            y_data = self._y[:self._current_size]
        if nb_value > y_data.shape[0]:
            padding_size = nb_value - y_data.shape[0]
            if y_data.shape[0] > 1:
                interval = np.median(np.diff(y_data))  # Calculate median interval
            else:
                interval = 1  # Default interval if not enough data points
            pad_start = y_data[0] - interval * padding_size if y_data.shape[0] > 0 else 0
            padding = np.arange(pad_start, y_data[0], interval, dtype=int) if y_data.shape[0] > 0 else np.zeros(padding_size, dtype=int)
            y_data = np.pad(y_data, (padding_size, 0), mode='constant', constant_values=pad_value)
            y_data[:padding_size] = padding
        return self._x, y_data, z_max