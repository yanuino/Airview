import serial
import time
import re
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import serial.tools.list_ports
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Max3D:
    def __init__(self, buffer_size=1, max_axis=0):
        self.buffer_size = buffer_size
        self.max_axis = max_axis
        self.max_buffer = None
        
    def append(self, data_array):
        max = np.max(data_array, axis=self.max_axis)  

        if self.max_buffer is None:
            self.max_buffer = max
            self.max_buffer = np.expand_dims(self.max_buffer, axis=0)
        else:
            self.max_buffer = np.append(self.max_buffer, [max], axis=0) 
        if self.max_buffer.shape[0] > self.buffer_size:
            self.max_buffer = self.max_buffer[-self.buffer_size:]

    def get3D(self):
        X, Y = np.meshgrid(np.arange(self.max_buffer.shape[1]), np.arange(self.max_buffer.shape[0]))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(X, Y, self.max_buffer/10, cmap='viridis')

        ax.set_ylabel('Series')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_zlabel('Signal Level (dB)')
        plt.show()

class Histogram3D:
    def __init__(self, buffer_size=1, bins=10, data_range=None):
        self.buffer_size = buffer_size
        self.bins = bins
        self.data_range = data_range
        self.histograms = None
        self.histogram_buffer = None

    def append(self, data_array):
        if self.histogram_buffer is None:
            if isinstance(self.bins, int):
                zsize = self.bins
            else:
                zsize = len(self.bins)-1
            self.histogram_buffer = np.zeros((0, data_array.shape[1], zsize))

        self.histograms = self.calculate_histograms(data_array)

        self.histogram_buffer = np.append(self.histogram_buffer, [self.histograms], axis=0)
        if self.histogram_buffer.shape[0] > self.buffer_size:
            self.histogram_buffer = self.histogram_buffer[-self.buffer_size:]

    def calculate_histograms(self, data):
        histograms = []
        for col in range(data.shape[1]):
            hist, _ = np.histogram(data[:, col],bins=self.bins, range=self.data_range)
            histograms.append(hist)
        return np.array(histograms)
    
    def get_histograms(self):
        if self.histograms is None:
            raise ValueError("Histograms have not been calculated yet.")
        return np.fliplr(np.rot90(self.histograms, axes=(1,0)))

class SpectrumAnalyzer(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(SpectrumAnalyzer, self).__init__(*args, **kwargs)
        self.x_range = np.arange(2399.0, 2485.5, 0.5)
        self.timerange = 900
        self.min_limit = -92
        self.full = np.full((1, len(self.x_range)), fill_value=-110, dtype=np.int8)
        self.lock = threading.Lock()
        self.worker = Worker(self)
        
        self.hist3d = Histogram3D(buffer_size=900, bins=np.arange(self.min_limit, 1))
        self.max3d = Max3D(buffer_size=100)

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

        pg.setConfigOption('imageAxisOrder', 'row-major')
        pg.setConfigOptions(antialias=True)
        
        hmajor_ticks = [(i, f"{self.x_range[i]:.1f}") for i in range(2, 173, 10)]
        hminor_ticks = [(i, '') for i in range(0, 173)]


        self.plotHM = self.win.addPlot(title='Heatmap')
        cmHM = pg.colormap.get('YlGnBu_r', source='matplotlib')
        self.img_hm = pg.ImageItem(image=self.full, levels=(-110, 0), lut=cmHM.getLookupTable(), enableAutoLevels=False)
        self.plotHM.addItem(self.img_hm)
        haxis = self.plotHM.getAxis('bottom')
        haxis.setTicks([hmajor_ticks, hminor_ticks])


        self.win.nextRow()
        self.plotHits = self.win.addPlot(title='Hits')

        cmHits = pg.colormap.get('YlGnBu_r', source='matplotlib')
        self.img_hits = pg.ImageItem(image=np.zeros((abs(self.min_limit), len(self.x_range)), dtype=np.int64), levels=(0, 90), lut=cmHits.getLookupTable(), enableAutoLevels=True, autoDownSample=True)
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

    def toggleButton(self):
        self.max3d.get3D()


    def get_last_elements(self, arr, number, pad_value=-110):
        padded_arr = np.pad(arr, ((max(0, number - arr.shape[0]), 0), (0, 0)), mode='constant', constant_values=pad_value)
        return padded_arr[-number:]

    def normalize_array(self, arr, min_val, max_val):
        return np.clip((arr - min_val) / (max_val - min_val), 0.2, 1.)

    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode = 'valid')
    
    @QtCore.pyqtSlot()
    def update(self):
        bins = np.arange(self.min_limit, 1)
        hits = np.zeros((abs(self.min_limit), len(self.x_range)), dtype=np.int64)

        with self.lock:
            z = self.full.copy()

        z_last90 = self.get_last_elements(z, 90)
        self.hist3d.append(z_last90)
        self.max3d.append(z_last90)

        max = np.max(z_last90, axis=0)
        avg = np.mean(z_last90, axis=0)

        window_size=5
        smooth = self.moving_average(z[-1], window_size)
        smooth_x = self.x_range[window_size -1:]

        hits = self.hist3d.get_histograms()

        self.c_values.setData(x=self.x_range, y=z[-1])
        self.c_avg.setData(x=self.x_range, y=avg)
        self.c_max.setData(x=self.x_range, y=max)
        # self.c_smooth.setData(x=smooth_x, y=smooth)
        self.img_hm.setImage(z_last90, autoLevels=False)
        self.img_hits.setImage(hits, autoLevels=True, autoDownsample=True)
      


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
                            i = np.where(self.parent.x_range == freq)[0][0]
                            values_int[i] = -98 if values_int[i] <= -90 else values_int[i]
                        with self.lock:
                            self.parent.full = np.append(self.parent.full[-self.parent.timerange + 1:], [values_int], axis=0)
                            # if self.parent.full.shape[0] > self.parent.timerange:
                            #     self.parent.full = self.parent.full[-self.parent.timerange:]
                            self.data_acquired.emit()

    def findport(self, desc):
        port = next((port.name for port in serial.tools.list_ports.comports() if desc.lower() in port.description.lower()), None)
        return port

if __name__ == '__main__':
    app = pg.mkQApp("Ubiquiti Airview Spectrum")
    mainWin = SpectrumAnalyzer()
    mainWin.show()
    pg.exec()
