import numpy as np
from numba import jit
import threading

class SpectrumDataStorage:
    def __init__(self, buffer_size=1):
        self.buffer_size = buffer_size
        self._max_window_size = 5
        self._smooth_window_size = 4
        
        self._x = np.arange(2399.0, 2485.5, 0.5)
        self._y = np.empty(0, dtype=int)
        self._z = np.empty((0, len(self._x)), dtype=float)
        self._z_max = np.empty((0, len(self._x)), dtype=float)
        self._z_smooth = np.empty((0, len(self._x[self._smooth_window_size -1:])), dtype=float)

        self.lock = threading.Lock()

    @property
    def x(self):
        with self.lock:
            return self._x.copy()
        
    @property
    def y(self):
        with self.lock:
            return self._y.copy()
        
    @property
    def z(self):
        with self.lock:
            return self._z.copy()

    @property
    def shape(self):
        with self.lock:
            return len(self._x), self._y.shape[0], self._z.shape[0], self._z.shape[1]
    
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
    def maxsmooth_window_sizewindow_size(self, value, recompute=False):
        if value < 1:
            raise ValueError("smooth window_size must be at least 1")
        with self.lock:
            self._smooth_window_size = value
            if recompute:
                self.__recompute_z_smooth()       

    #@jit(nopython=True)
    def __moving_average(self, data):
        window = np.ones(self._smooth_window_size) / self._smooth_window_size
        return np.convolve(data, window, mode = 'valid')
    
    def __recompute_z_max(self):
        z_max_list = []
        for i in range(len(self._z)):
            if i < self._max_window_size:
                max_z = np.max(self._z[:i + 1], axis=0)
            else:
                max_z = np.max(self._z[i - self._max_window_size + 1:i + 1], axis=0)
            z_max_list.append(max_z)
        self._z_max = np.vstack(z_max_list)

    def __recompute_z_smooth(self):
        smoothed_data = []
        for i in range(len(self._z)):
            smoothed_data.append(self.__moving_average(self._z[i]))
        self._z_smooth = np.vstack(smoothed_data)

    def append(self, y, z):
        if len(z) != len(self._x):
            raise ValueError("Length of z must be equal to length of x")
        
        with self.lock:
            if len(self._y) < self.buffer_size:
                self._y = np.append(self._y, y)
                self._z = np.append(self._z, [z], axis=0)
                self._z_smooth = np.append(self._z_smooth, [self.__moving_average(z)], axis=0)
            else:
                self._y = np.roll(self._y, -1)
                self._z = np.roll(self._z, -1, axis=0)
                self._z_smooth = np.roll(self._z_smooth, -1, axis=0)
                self._y[-1] = y
                self._z[-1] = z
                self._z_smooth[-1] = self.__moving_average(z)

            # Compute z_max
            if len(self._z) <= self._max_window_size:
                max_z = np.max(self._z, axis=0)
            else:
                max_z = np.max(self._z[-self._max_window_size:], axis=0)

            if len(self._z_max) < self.buffer_size:
                self._z_max = np.append(self._z_max, [max_z], axis=0)
            else:
                self._z_max = np.roll(self._z_max, -1, axis=0)
                self._z_max[-1] = max_z

    def get(self):
        with self.lock:
            return self._z[-1]
        
    def get_smooth(self):
        with self.lock:
            return self._x[self._smooth_window_size -1:], self._z_smooth[-1]
        
    def get_max(self, number=None):
        with self.lock:
            if number is None:
                return np.max(self._z, axis=0)

            if number > self.buffer_size:
                raise ValueError("number cannot be greater than buffer_size")
            
            return np.max(self._z[-number:], axis=0)

    def get_mean(self, number=None):
        with self.lock:
            if number is None:
                return np.mean(self._z, axis=0)

            if number > self.buffer_size:
                raise ValueError("number cannot be greater than buffer_size")
            
            return np.mean(self._z[-number:], axis=0)


    def get_data(self, nb_value=None, pad_value=0):
        with self.lock:
            if nb_value is None:
                return self._z
            
            if nb_value > self.buffer_size:
                raise ValueError("nb_value cannot be greater than buffer_size")
            
            data = self._z if nb_value is None else self._z[-nb_value:]
            if nb_value > data.shape[0]:
                padding = ((nb_value - data.shape[0], 0), (0, 0))
                data = np.pad(data, padding, mode='constant', constant_values=pad_value)
            return data

    def histogram(self, nb_value=None, bins=10, data_range=None):
        with self.lock:
            hist = []
            bin_edges = None
            
            data = self._z if nb_value is None else self._z[-nb_value:]
            for i in range(data.shape[1]):
                h, edges = np.histogram(data[:, i], bins=bins, range=data_range)
                hist.append(h)
                if bin_edges is None:
                    bin_edges = edges
        return np.fliplr(np.rot90(np.array(hist), axes=(1,0))), bin_edges

    def __get_maxdata(self, nb_value=None, pad_value=0):
        if nb_value is None:
            return self._z_max
        
        if nb_value > self.buffer_size:
            raise ValueError("nb_value cannot be greater than buffer_size")
        
        maxdata = self.self.z_max if nb_value is None else self.self.z_max[-nb_value:]
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
        
        if nb_value > y_data.shape[0]:
            padding_size = nb_value - y_data.shape[0]
            if y_data.shape[0] > 1:
                interval = np.median(np.diff(y_data))  # Calculate median interval
            else:
                interval = 1  # Default interval if not enough data points

            pad_start = y_data[0] - interval * padding_size
            padding = np.arange(pad_start, y_data[0], interval, dtype=int)
            y_data = np.pad(y_data, (padding_size, 0), mode='constant', constant_values=pad_value)
            y_data[:padding_size] = padding

        return self._x, y_data, z_data
        
    def get_max3D(self, nb_value=None, pad_value=0):
        z_max = self.__get_maxdata(nb_value, pad_value)
        if nb_value is None:
            nb_value = z_max.shape[0]

        if nb_value > self.buffer_size:
            raise ValueError("nb_value cannot be greater than buffer_size")
        
        if nb_value > y_data.shape[0]:
            padding_size = nb_value - y_data.shape[0]
            if y_data.shape[0] > 1:
                interval = np.median(np.diff(y_data))  # Calculate median interval
            else:
                interval = 1  # Default interval if not enough data points

            pad_start = y_data[0] - interval * padding_size
            padding = np.arange(pad_start, y_data[0], interval, dtype=int)
            y_data = np.pad(y_data, (padding_size, 0), mode='constant', constant_values=pad_value)
            y_data[:padding_size] = padding

        return self._x, y_data, z_max  