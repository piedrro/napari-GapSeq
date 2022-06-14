"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

from qtpy.QtWidgets import (QWidget,QVBoxLayout,QTabWidget,QCheckBox,QLabel,QLineEdit,QFileDialog,
                            QComboBox,QPushButton,QProgressBar,QTextEdit,QSlider,QSpinBox, QSpacerItem, QSizePolicy)
from qtpy.QtCore import (QObject,QRunnable,QThreadPool)
from PyQt5.QtCore import pyqtSignal,pyqtSlot
import sys
from functools import partial
import os
import traceback
import napari
import numpy as np
import time
import cv2
import pandas as pd
from glob2 import glob
import tifffile
import cv2 as cv
from skimage import exposure
# import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import time


plt.style.use('dark_background')

from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari


def normalize99(X):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """

    if np.max(X) > 0:
        X = X.copy()
        v_min, v_max = np.percentile(X[X != 0], (1, 99))
        X = exposure.rescale_intensity(X, in_range=(v_min, v_max))

    return X


def rescale01(x):
    """ normalize image from 0 to 1 """

    if np.max(x) > 0:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

    return x


def find_contours(img):
    # finds contours of shapes, only returns the external contours of the shapes

    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours


class GapSeqTabWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: napari.Viewer):

        super().__init__()

        from napari_gapseq.gapseq_ui import Ui_TabWidget

        application_path = os.path.dirname(sys.executable)
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.form = Ui_TabWidget()
        self.akseg_ui = QTabWidget()
        self.form.setupUi(self.akseg_ui)

        self.graph_container = QWidget()
        self.canvas = FigureCanvasQTAgg()
        self.canvas.figure.set_tight_layout(True)
        self.canvas.figure.patch.set_facecolor("#262930")
        self.graph_container.setLayout(QVBoxLayout())
        self.graph_container.setMaximumHeight(300)
        self.graph_container.layout().addWidget(self.canvas)

        self.akseg_ui.addTab(self.graph_container, "Plots")

        self.layout().addWidget(self.akseg_ui)

        #events

        self.import_localisations = self.findChild(QPushButton, "import_localisations")
        self.import_image = self.findChild(QPushButton, "import_image")

        self.import_localisations.clicked.connect(self.import_localisation_file)
        self.import_image.clicked.connect(self.import_image_file)

        self.canvas.mpl_connect("button_press_event", self.on_press)

        self.viewer.dims.events.current_step.connect(self.update_plot)

        self.current_coord = None

    def on_press(self, event):

        if event.xdata != None:

            frame_int = int(event.xdata)

            current_dims = self.viewer.dims.current_step

            new_dims = tuple([frame_int, *list(current_dims)[1:]])

            self.viewer.dims.current_step = new_dims


    def import_localisation_file(self):

        # desktop = os.path.expanduser("~/Desktop")
        # path, filter = QFileDialog.getOpenFileName(self, "Open Files", desktop, "Files (*)")

        path = r"C:/napari-gapseq/src/napari_gapseq/dev/20220527_27thMay2022GAP36A/27thMay2022GAPSeq4onebyonesubstrategAPGS8FRETfoursea50nMconce2_GAPSeqonebyoneGAP36AL532Exp200.tif"

        image = tifffile.imread(path)

        image = image[:, :, : image.shape[2] // 2]

        img = np.max(image, axis=0)

        img = normalize99(img)
        img = rescale01(img) * 255
        img = img.astype(np.uint8)

        img = cv.fastNlMeansDenoising(img, h=30, templateWindowSize=5, searchWindowSize=31)

        _, mask = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

        contours = find_contours(mask)

        polygons = []

        for i in range(len(contours)):

            try:

                cnt = contours[i]

                x, y, w, h = cv.boundingRect(cnt)

                M = cv.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                polygon = np.array([[cy - 2, cx - 2], [cy + 2, cx + 2]])

                polygons.append(polygon)

                # cv.rectangle(img, (cx - 2, cy - 2), (cx + 2, cy + 2), (0, 128, 255), 2)

            except:
                pass


        self.localisation_layer = self.viewer.add_image(image, name="Localisations")
        self.localisation_layer.mouse_drag_callbacks.append(partial(self.select_localisation,mode="click"))

        self.box_layer = self.viewer.add_shapes(polygons, shape_type='Rectangle', edge_width=1, edge_color='red', face_color=[0,0,0,0], opacity=0.3)
        self.box_layer.mouse_drag_callbacks.append(partial(self.select_localisation,mode="click"))

        self.viewer.layers.selection.events.changed.connect(partial(self.select_localisation,mode="layer_change"))

    def import_image_file(self):

        path = r"C:/napari-gapseq/src/napari_gapseq/dev/20220527_27thMay2022GAP36A/27thMay2022GAPSeq4onebyonesubstrategAPGS8FRETfoursea25nM_GAPSeqonebyoneGS83TL638Exp200.tif"

        gap_codes = ["A","T","C","G"]
        layer_names = []
        for i in range(len(gap_codes)):

            index = len(path) - 15
            path = path[:index] + gap_codes[i] + path[index + 1:]

            image = tifffile.imread(path)

            image = image[:, :, image.shape[2] // 2 : ]

            layer_name = f"SEQ-{gap_codes[i]}"
            layer_names.append(layer_name)

            setattr(self, layer_name, self.viewer.add_image(image, name=layer_name))
            self.viewer.layers[layer_name].mouse_drag_callbacks.append(partial(self.select_localisation, mode="click"))


    def select_localisation(self, viewer=None, event = None,  mode = "click"):

        if mode == "click":
            selected_layer = self.viewer.layers.selection.active
            data_coordinates = selected_layer.world_to_data(event.position)
            coord = np.round(data_coordinates).astype(int)
            self.current_coord = coord
        if mode == "layer_change":
            coord = self.current_coord

        if coord is not None:

            if len(coord) > 2:

                coord = coord[1:]

            polygons = self.box_layer.data

            box_colour = self.box_layer.get_value(coord)[0]

            if box_colour is not None:

                [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = polygons[box_colour]

                current_frame = int(self.viewer.dims.current_step[0])

                localisation_data = []

                for layer in self.viewer.layers.selection:

                    layer_name = str(layer)

                    if layer_name != "polygons":

                        spot = layer.data[:, int(y1):int(y2), int(x1):int(x2)]
                        data = np.mean(spot, axis=(1, 2))

                        localisation_data.append({'layer_name':layer_name, 'data':data, 'spot':spot[current_frame]})

                self.plot(self.canvas, plot_data = localisation_data)


    def plot(self, canvas, plot_data, xlabel = "Frame", ylabel = "Intensity"):

        canvas.figure.clf()

        maximum_width = 200*len(plot_data)

        if maximum_width != 0:
            self.graph_container.setMaximumHeight(maximum_width)

        for i in range(len(plot_data)):

            plot_index = int(f"{len(plot_data)}1{i+1}")
            # spot_index = int(f"{len(plot_data)}2{i+2}")

            axes = canvas.figure.add_subplot(plot_index)
            layer_name = plot_data[i]["layer_name"]
            spot = plot_data[i]["spot"]

            y = plot_data[i]["data"]
            x = np.arange(len(y))

            axes.set_facecolor("#262930")
            axes.plot(x, y, label = layer_name)
            axes.legend()

            ymin, ymax = axes.get_ylim()
            current_frame = int(self.viewer.dims.current_step[0])

            axes.plot([current_frame,current_frame],[ymin, ymax], color = "red", label = "Frame")

            # spot_axes = canvas.figure.add_subplot(spot_index)
            # spot_axes.imshow(spot)

        self.canvas.figure.tight_layout()
        canvas.draw()

    def update_plot(self):

        allaxes = self.canvas.figure.get_axes()

        if len(allaxes) > 0:

            current_frame = int(self.viewer.dims.current_step[0])

            for ax in allaxes:

                # print(ax.get_gridspec())

                vertical_line = ax.lines[1]
                vertical_line.set_xdata([current_frame, current_frame])
                self.canvas.draw()
                self.canvas.flush_events()


