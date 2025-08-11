import serial
import time
import re
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import serial.tools.list_ports
import threading

from spectrumdata import SpectrumDataStorage   

class SpectrumAnalyzer(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(SpectrumAnalyzer, self).__init__(*args, **kwargs)
        self.timerange = 900
        self.values = 90
        self.raw = True
        self.min_limit = -92
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
        
        # Use frequency values for ticks
        x = self.spectrum.x
        hmajor_ticks = [(freq, f"{freq:.0f}") for freq in x if freq % 10 == 0]
        hminor_ticks = [(freq, '') for freq in x]



        self.plotHM = self.win.addPlot(title='Heatmap')
        cmHM = pg.colormap.get('YlGnBu_r', source='matplotlib')
        self.img_hm = pg.ImageItem(image=self.spectrum._z, levels=(-110, 0), lut=cmHM.getLookupTable(), enableAutoLevels=False)
        # Set the image rectangle to match frequency axis
        self.img_hm.setRect(QtCore.QRectF(x[0], 0, x[-1] - x[0], self.spectrum._z.shape[0]))
        self.plotHM.addItem(self.img_hm)
        haxis = self.plotHM.getAxis('bottom')
        haxis.setTicks([hmajor_ticks, hminor_ticks])


        self.win.nextRow()
        self.plotHits = self.win.addPlot(title='Hits')

        cmHits = pg.colormap.get('YlGnBu_r', source='matplotlib')
        self.img_hits = pg.ImageItem(image=np.zeros((abs(self.min_limit), self.spectrum.shape[0]), dtype=np.int64), levels=(0, 90), lut=cmHits.getLookupTable(), enableAutoLevels=True, autoDownSample=True)
        # Set the image rectangle for Hits as well
        self.img_hits.setRect(QtCore.QRectF(x[0], 0, x[-1] - x[0], abs(self.min_limit)))
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
        self.worker_thread = QtCore.QThread(self)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.acquire)
        self.worker.data_acquired.connect(self.update, QtCore.Qt.QueuedConnection)
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
        x = self.spectrum.x
        hits, _ = self.spectrum.histogram(nb_value=self.values, bins=np.arange(self.min_limit, 1))
        if self.raw:
            self.c_values.setData(x=x, y=self.spectrum.get())
        else:
            smooth_x, smooth = self.spectrum.get_smooth()
            self.c_values.setData(x=smooth_x, y=smooth)
        self.c_avg.setData(x=x, y=self.spectrum.get_mean(self.values))
        self.c_max.setData(x=x, y=self.spectrum.get_max(self.values))
        self.img_hm.setImage(self.spectrum.get_data(self.values, -110), autoLevels=False)
        self.img_hits.setImage(hits, autoLevels=True, autoDownSample=True)

class Worker(QtCore.QObject):
    data_acquired = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
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
                        # for freq in (2400.0, 2412.0, 2424.0, 2436.0, 2448.0, 2460.0, 2484.0):
                        #     i = np.where(self.parent.spectrum.x == freq)[0][0]
                        #     values_int[i] = -98 if values_int[i] <= -90 else values_int[i]
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
