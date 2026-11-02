import time

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from pyqtgraph.Qt import QtCore, QtWidgets

from spectrumdata import SpectrumDataStorage


class SpectrumAnalyzer(QtWidgets.QMainWindow):
    """Display live spectrum measurements in a Qt window."""

    def __init__(self, *args, **kwargs):
        """Initialize the analyzer window, data storage, and acquisition worker."""
        super(SpectrumAnalyzer, self).__init__(*args, **kwargs)
        self.timerange = 900
        self.values = 90
        self.raw = True
        self.min_limit = -92
        self.spectrum = SpectrumDataStorage(self.timerange)
        self.worker = Worker(self.spectrum)

        self.initUI()
        self.initWorker()

    def initUI(self):
        """Create the plots and controls in the main window."""
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

        pg.setConfigOption("imageAxisOrder", "row-major")
        pg.setConfigOptions(antialias=True)

        # Use frequency values for ticks
        x = self.spectrum.x
        hmajor_ticks = [(freq, f"{freq:.0f}") for freq in x if freq % 10 == 0]
        hminor_ticks = [(freq, "") for freq in x]

        self.plotHM = self.win.addPlot(title="Heatmap")
        cmHM = pg.colormap.get("YlGnBu_r", source="matplotlib")
        self.img_hm = pg.ImageItem(
            image=self.spectrum._z,
            levels=(-110, 0),
            lut=cmHM.getLookupTable(),
            enableAutoLevels=False,
        )
        # Set the image rectangle to match frequency axis
        self.img_hm.setRect(
            QtCore.QRectF(x[0], 0, x[-1] - x[0], self.spectrum._z.shape[0])
        )
        self.plotHM.addItem(self.img_hm)
        haxis = self.plotHM.getAxis("bottom")
        haxis.setTicks([hmajor_ticks, hminor_ticks])

        self.win.nextRow()
        self.plotHits = self.win.addPlot(title="Hits")

        cmHits = pg.colormap.get("YlGnBu_r", source="matplotlib")
        self.img_hits = pg.ImageItem(
            image=np.zeros(
                (abs(self.min_limit), self.spectrum.shape[0]), dtype=np.int64
            ),
            levels=(0, 90),
            lut=cmHits.getLookupTable(),
            enableAutoLevels=True,
            autoDownSample=True,
        )
        # Set the image rectangle for Hits as well
        self.img_hits.setRect(QtCore.QRectF(x[0], 0, x[-1] - x[0], abs(self.min_limit)))
        self.plotHits.addItem(self.img_hits)
        haxis = self.plotHits.getAxis("bottom")
        haxis.setTicks([hmajor_ticks, hminor_ticks])
        vmajor_ticks = [
            (i, f"{np.arange(self.min_limit, 1)[i]:d}") for i in range(2, 93, 10)
        ]
        vminor_ticks = [(i, "") for i in range(0, 93, 2)]
        vaxis = self.plotHits.getAxis("left")
        vaxis.setTicks([vmajor_ticks, vminor_ticks])

        self.win.nextRow()
        self.plotLvl = self.win.addPlot(title="Levels")
        self.plotLvl.setRange(yRange=(-100, -20), update=True, disableAutoRange=True)
        self.plotLvl.getAxis("left").setTickSpacing(major=10, minor=1)
        self.plotLvl.showGrid(x=False, y=True, alpha=0.3)
        self.c_values = self.plotLvl.plot(name="Current", pen="y")
        self.c_max = self.plotLvl.plot(
            name="Maximum", pen="b", brush=(50, 50, 200, 100), fillLevel=-110
        )
        self.c_avg = self.plotLvl.plot(
            name="Average", pen="g", brush=(50, 200, 50, 100), fillLevel=-110
        )

        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.plotLvl.addItem(self.vLine, ignoreBounds=True)

        self.freqTxt = pg.TextItem("freq")
        self.freqTxt.setParentItem(self.vLine)
        self.vbLvl = self.plotLvl.vb
        self.plotLvl.scene().sigMouseMoved.connect(self.mouseMoved)

        self.rawRadio = QtWidgets.QRadioButton("raw")
        self.smoothRadio = QtWidgets.QRadioButton("smooth")
        self.rawRadio.setChecked(True)
        self.rawRadio.toggled.connect(self.setPlotMode)
        self.bottom_layout.addWidget(self.rawRadio, 0, 1)
        self.bottom_layout.addWidget(self.smoothRadio, 0, 2)

        lblspin = QtWidgets.QLabel("Values")
        spin = pg.SpinBox(
            value=self.values,
            int=True,
            dec=True,
            bounds=[1, None],
            minStep=1,
            step=1,
            decimals=4,
        )
        spin.sigValueChanged.connect(self.setValues)
        self.bottom_layout.addWidget(lblspin, 0, 3)
        self.bottom_layout.addWidget(spin, 0, 4)

        self.toggle_button = QtWidgets.QPushButton("Test")
        self.toggle_button.clicked.connect(self.toggleButton)
        self.layout.addWidget(self.toggle_button)

    def initWorker(self):
        """Start the worker thread that acquires spectrum samples."""
        self.worker_thread = QtCore.QThread(self)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.acquire)
        self.worker.data_acquired.connect(self.update_plots)
        self.worker_thread.start()

    def closeEvent(self, event):
        """Stop the acquisition worker before closing the window.

        Args:
            event: Qt close event to accept after the worker stops.
        """
        self.worker.stop_thread = True
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

    def mouseMoved(self, evt):
        """Update the frequency marker when the pointer is over the level plot.

        Args:
            evt: Scene position of the pointer.
        """
        pos = evt
        if self.plotLvl.sceneBoundingRect().contains(pos):
            mousePoint = self.vbLvl.mapSceneToView(pos)
            index = np.round(mousePoint.x() * 2) / 2
            self.freqTxt.setText("%0.1f" % index)
            self.freqTxt.setAnchor((0.0, -3.0))
            self.vLine.setPos(index)

    def setPlotMode(self):
        """Set the plot mode from the raw/smoothed radio buttons."""
        self.raw = self.rawRadio.isChecked()

    def setValues(self, sb):
        """Update the number of recent samples shown in the plots.

        Args:
            sb: Spin box containing the requested sample count.
        """
        self.values = sb.value()

    def setSmoothWindow(self, sb):
        """Handle a change to the smoothing-window control.

        Args:
            sb: Control that triggered the change.
        """
        with pg.BusyCursor():
            pass

    def toggleButton(self):
        """Handle a click of the test button."""
        self.wgl.show()
        pass

    def normalize_array(self, arr, min_val, max_val):
        """Normalize an array and clip its values to the range 0.2 through 1.0."""
        return np.clip((arr - min_val) / (max_val - min_val), 0.2, 1.0)

    @QtCore.Slot()
    def update_plots(self):
        """Refresh the plots with the latest spectrum data.

        Nothing is drawn while the window is minimized; data keeps accumulating.
        """
        if self.isMinimized():
            return
        x = self.spectrum.x
        hits, _ = self.spectrum.histogram(
            nb_value=self.values, bins=np.arange(self.min_limit, 1), clip=True
        )
        if self.raw:
            self.c_values.setData(x=x, y=self.spectrum.get(), skipFiniteCheck=True)
        else:
            smooth_x, smooth = self.spectrum.get_smooth()
            self.c_values.setData(x=smooth_x, y=smooth, skipFiniteCheck=True)
        self.c_avg.setData(
            x=x, y=self.spectrum.get_mean(self.values), skipFiniteCheck=True
        )
        self.c_max.setData(
            x=x, y=self.spectrum.get_max(self.values), skipFiniteCheck=True
        )
        self.img_hm.setImage(
            self.spectrum.get_data(self.values, -110), autoLevels=False
        )
        # Edge rows hold clipped values and would otherwise dominate the colour scale.
        hits_max = max(int(hits[1:-1].max(initial=0)), 1)
        self.img_hits.setImage(
            hits, levels=(0, hits_max), autoLevels=False, autoDownSample=True
        )


class Worker(QtCore.QObject):
    """Acquire spectrum samples from the analyzer's serial connection."""

    data_acquired = QtCore.Signal()

    def __init__(self, spectrum):
        """Initialize the worker with the storage receiving acquired samples.

        Args:
            spectrum: Spectrum storage that receives acquired samples.
        """
        super().__init__()
        self.spectrum = spectrum
        self.stop_thread = False

    def acquire(self):
        """Read spectrum scans from the analyzer and emit data-acquired signals."""
        port = self.findport("AirView")
        with serial.Serial(port, 115200, timeout=1) as ser:
            ser.write(b"\ngdi\n\ninit\n\nbs\n")

            while not self.stop_thread:
                line = ser.readline()
                # A read timeout can return a partial line, which has no newline.
                if not (line.startswith(b"scan|") and line.endswith(b"\n")):
                    continue
                try:
                    values_int = np.array(
                        list(map(int, line.split(b",", 1)[1].split())), dtype=np.int64
                    )
                    self.spectrum.append(time.time_ns(), values_int)
                except IndexError, ValueError:
                    continue
                self.data_acquired.emit()

    def findport(self, desc):
        """Find the name of the first serial port matching a description.

        Args:
            desc: Case-insensitive substring to search for in port descriptions.

        Returns:
            The matching port name, or None if no port matches.
        """
        port = next(
            (
                port.name
                for port in serial.tools.list_ports.comports()
                if desc.lower() in port.description.lower()
            ),
            None,
        )
        return port


if __name__ == "__main__":
    app = pg.mkQApp("Ubiquiti Airview Spectrum")
    mainWin = SpectrumAnalyzer()
    mainWin.show()
    pg.exec()
