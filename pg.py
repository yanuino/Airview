import serial
import time
import re
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import serial.tools.list_ports
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

    def __moving_average(self, data):
        return np.convolve(data, np.ones(self._smooth_window_size) / self._smooth_window_size, mode = 'valid')
    
    def __recompute_z_max(self):
        self._z_max = np.empty((0, len(self._x)), dtype=float)
        for i in range(len(self._z)):
            if i < self._max_window_size:
                max_z = np.max(self._z[:i + 1], axis=0)
            else:
                max_z = np.max(self._z[i - self._max_window_size + 1:i + 1], axis=0)
            self._z_max = np.append(self._z_max, [max_z], axis=0)

    def __recompute_z_smooth(self):
        self._z_smooth = np.empty((0, len(self._x[self._smooth_window_size -1:])), dtype=float)
        for i in range(len(self._z)):
            self._z_smooth = np.append(self._z_smooth, [self.__moving_average(self._z[i])], axis=0)

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


class SpectrumAnalyzer(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(SpectrumAnalyzer, self).__init__(*args, **kwargs)
        self.timerange = 900
        self.values = 90
        self.raw = True
        self.min_limit = -92
        self.lock = threading.Lock()
        self.worker = Worker(self)

        self.spectrum = SpectrumDataStorage(self.timerange)        

        self.initUI()
        self.initWorker()

    def initUI(self):
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle("Ubiquiti Airview Spectrum")
        self.resize(1000, 600)
        
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win)
        self.bottom_widget = QtWidgets.QWidget()
        self.bottom_layout = QtWidgets.QGridLayout(self.bottom_widget)
        self.layout.addWidget(self.bottom_widget)

        pg.setConfigOption('imageAxisOrder', 'row-major')
        pg.setConfigOptions(antialias=True)
        
        hmajor_ticks = [(i, f"{self.spectrum._x[i]:.1f}") for i in range(2, 173, 10)]
        hminor_ticks = [(i, '') for i in range(0, 173)]


        self.plotHM = self.win.addPlot(title='Heatmap')
        cmHM = pg.colormap.get('YlGnBu_r', source='matplotlib')
        self.img_hm = pg.ImageItem(image=self.spectrum._z, levels=(-110, 0), lut=cmHM.getLookupTable(), enableAutoLevels=False)
        self.plotHM.addItem(self.img_hm)
        haxis = self.plotHM.getAxis('bottom')
        haxis.setTicks([hmajor_ticks, hminor_ticks])


        self.win.nextRow()
        self.plotHits = self.win.addPlot(title='Hits')

        cmHits = pg.colormap.get('YlGnBu_r', source='matplotlib')
        self.img_hits = pg.ImageItem(image=np.zeros((abs(self.min_limit), self.spectrum.shape[0]), dtype=np.int64), levels=(0, 90), lut=cmHits.getLookupTable(), enableAutoLevels=True, autoDownSample=True)
        self.plotHits.addItem(self.img_hits)
        haxis = self.plotHits.getAxis('bottom')
        haxis.setTicks([hmajor_ticks, hminor_ticks])
        vmajor_ticks = [(i, f"{np.arange(self.min_limit, 1)[i]:d}") for i in range(2, 93, 10)]
        vminor_ticks = [(i, '') for i in range(0, 93, 2)]
        vaxis = self.plotHits.getAxis('left')
        vaxis.setTicks([vmajor_ticks, vminor_ticks])

        self.win.nextRow()
        self.plotLvl = self.win.addPlot(title='Levels')
        self.plotLvl.setRange(yRange=(-100, -20), update=True, disableAutoRange=True)
        self.plotLvl.getAxis('left').setTickSpacing(major=10, minor=1)
        self.plotLvl.showGrid(x=False, y=True, alpha=0.3)
        self.c_values = self.plotLvl.plot(name='Current', pen='y')
        self.c_max = self.plotLvl.plot(name='Maximum', pen='b', brush=(50, 50, 200, 100), fillLevel=-110)
        self.c_avg = self.plotLvl.plot(name='Average', pen='g', brush=(50, 200, 50, 100), fillLevel=-110)

        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.plotLvl.addItem(self.vLine, ignoreBounds=True)

        self.freqTxt = pg.TextItem("freq")
        self.freqTxt.setParentItem(self.vLine)
        self.vbLvl = self.plotLvl.vb
        self.plotLvl.scene().sigMouseMoved.connect(self.mouseMoved)

        self.rawRadio = QtWidgets.QRadioButton('raw')
        self.smoothRadio = QtWidgets.QRadioButton('smooth')
        self.rawRadio.setChecked(True)
        self.rawRadio.toggled.connect(self.setPlotMode)
        self.bottom_layout.addWidget(self.rawRadio, 0, 1)
        self.bottom_layout.addWidget(self.smoothRadio, 0, 2)

        lblspin = QtWidgets.QLabel('Values')
        spin = pg.SpinBox(value=self.values, int=True, dec=True, bounds=[1, None], minStep=1, step=1, decimals=4)
        spin.sigValueChanged.connect(self.setValues)
        self.bottom_layout.addWidget(lblspin, 0, 3)
        self.bottom_layout.addWidget(spin, 0, 4)

        self.toggle_button = QtWidgets.QPushButton("Test")
        self.toggle_button.clicked.connect(self.toggleButton)
        self.layout.addWidget(self.toggle_button)

    def initWorker(self):
        self.worker_thread = QtCore.QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.acquire)
        self.worker.data_acquired.connect(self.update)
        self.worker_thread.start()

    def closeEvent(self, event):
        self.worker.stop_thread = True
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()
    
    def mouseMoved(self, evt):
        pos = evt
        if self.plotLvl.sceneBoundingRect().contains(pos):
            mousePoint = self.vbLvl.mapSceneToView(pos)
            index = np.round(mousePoint.x() * 2) /2
            self.freqTxt.setText('%0.1f' % index)
            self.freqTxt.setAnchor((0.0, -3.0))
            self.vLine.setPos(index)

    def setPlotMode(self):
        self.raw = self.rawRadio.isChecked()

    def setValues(self, sb):
        self.values = sb.value()

    def setSmoothWindow(self,sb):
        with pg.BusyCursor():
            pass

    def toggleButton(self):
        self.wgl.show()
        pass

    def normalize_array(self, arr, min_val, max_val):
        return np.clip((arr - min_val) / (max_val - min_val), 0.2, 1.)

    @QtCore.pyqtSlot()
    def update(self):

        x= self.spectrum.x

        hits, _ = self.spectrum.histogram(nb_value=self.values, bins=np.arange(self.min_limit, 1))

        if self.raw:
            self.c_values.setData(x=x, y=self.spectrum.get())
        else:
            smooth_x, smooth = self.spectrum.get_smooth()
            self.c_values.setData(x=smooth_x, y=smooth)

        self.c_avg.setData(x=x, y=self.spectrum.get_mean(self.values))
        self.c_max.setData(x=x, y=self.spectrum.get_max(self.values))
        self.img_hm.setImage(self.spectrum.get_data(self.values, -110), autoLevels=False)
        self.img_hits.setImage(hits, autoLevels=True, autoDownsample=True)

        self.sp.setData(z=self.spectrum.z)
      
class Worker(QtCore.QObject):
    data_acquired = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.lock = threading.Lock()
        self.stop_thread = False

    def acquire(self):
        port = self.findport("AirView")
        with serial.Serial(port, 115200, timeout=1) as ser:
            ser.write(b"\x0A")
            ser.write(b"gdi")
            ser.write(b"\x0A")
            ser.write(b"\x0A")
            ser.write(b"init")
            ser.write(b"\x0A")
            ser.write(b"\x0A")
            ser.write(b"bs")
            ser.write(b"\x0A")

            while not self.stop_thread:
                line = ser.readline()
                m = re.findall(b'scan\\|\\d+,(.*)\\n', line)
                if m:
                    for n in m:
                        values = n.split()
                        values_int = np.array([int(v.decode('utf-8')) for v in values])
                        for freq in (2400.0, 2412.0, 2424.0, 2436.0, 2448.0, 2460.0, 2484.0):
                            i = np.where(self.parent.spectrum.x == freq)[0][0]
                            values_int[i] = -98 if values_int[i] <= -90 else values_int[i]
                        timestamp = time.time_ns()
                        self.parent.spectrum.append(timestamp, values_int)
                        self.data_acquired.emit()

    def findport(self, desc):
        port = next((port.name for port in serial.tools.list_ports.comports() if desc.lower() in port.description.lower()), None)
        return port

if __name__ == '__main__':
    app = pg.mkQApp("Ubiquiti Airview Spectrum")
    mainWin = SpectrumAnalyzer()
    mainWin.show()
    pg.exec()
