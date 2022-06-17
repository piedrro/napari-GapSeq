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
import json

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
        from napari_gapseq.gapseq_plot_ui import Ui_PlotUI

        application_path = os.path.dirname(sys.executable)
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.form = Ui_TabWidget()
        self.akseg_ui = QTabWidget()
        self.form.setupUi(self.akseg_ui)
        self.layout().addWidget(self.akseg_ui)

        #register QWidgets/Controls
        self.localisation_channel = self.findChild(QComboBox,"localisation_channel")
        self.localisation_import_image = self.findChild(QPushButton,"localisation_import_image")
        self.localisation_threshold = self.findChild(QSlider,"localisation_threshold")
        self.localisation_area_min = self.findChild(QSlider,"localisation_area_min")
        self.localisation_area_max = self.findChild(QSlider,"localisation_area_max")
        self.localisation_bbox_size = self.findChild(QSlider,"localisation_bbox_size")
        self.localisation_threshold_label = self.findChild(QLabel,"localisation_threshold_label")
        self.localisation_area_min_label = self.findChild(QLabel,"localisation_area_min_label")
        self.localisation_area_max_label = self.findChild(QLabel,"localisation_area_max_label")
        self.localisation_bbox_size_label = self.findChild(QLabel,"localisation_bbox_size_label")
        self.localisation_detect = self.findChild(QPushButton,"localisation_detect")
        self.localisation_live_detection = self.findChild(QCheckBox,"localisation_live_detection")
        self.load_dev = self.findChild(QPushButton,"load_dev")
        self.plot_compute = self.findChild(QPushButton,"plot_compute")
        self.plot_compute_progress = self.findChild(QProgressBar, "plot_compute_progress")
        self.plot_mode = self.findChild(QComboBox,"plot_mode")
        self.plot_localisation_number = self.findChild(QSlider,"plot_localisation_number")
        self.plot_frame_number = self.findChild(QSlider,"plot_frame_number")
        self.plot_localisation_class = self.findChild(QComboBox,"plot_localisation_class")
        self.plot_localisation_classify = self.findChild(QPushButton,"plot_localisation_classify")
        self.plot_localisation_filter = self.findChild(QComboBox,"plot_localisation_filter")
        self.plot_localisation_focus = self.findChild(QCheckBox,"plot_localisation_focus")
        self.graph_container = self.findChild(QWidget,"graph_container")
        self.gapseq_export_data = self.findChild(QPushButton,"gapseq_export_data")
        self.gapseq_export_at_import = self.findChild(QCheckBox,"gapseq_export_at_import")
        self.gapseq_import_localisations = self.findChild(QPushButton,"gapseq_import_localisations")
        self.gapseq_import_all = self.findChild(QPushButton, "gapseq_import_all")
        self.gapseq_export_traces = self.findChild(QPushButton, "gapseq_export_traces")

        self.graph_container.setLayout(QVBoxLayout())
        # self.graph_container.setMaximumHeight(500)
        self.graph_container.setMinimumWidth(100)

        self.canvas = FigureCanvasQTAgg()
        self.canvas.figure.set_tight_layout(True)
        self.canvas.figure.patch.set_facecolor("#262930")
        self.graph_container.layout().addWidget(self.canvas)

        self.image_import_channel = self.findChild(QComboBox,"image_import_channel")
        self.image_gap_code = self.findChild(QComboBox,"image_gap_code")
        self.image_sequence_code = self.findChild(QComboBox, "image_sequence_code")
        self.import_image = self.findChild(QPushButton,"import_image")

        self.import_localisations = self.findChild(QPushButton, "import_localisations")
        self.import_image = self.findChild(QPushButton, "import_image")

        # self.import_localisations.clicked.connect(self.import_localisation_file)
        # self.import_image.clicked.connect(self.import_image_file)

        # self.canvas.mpl_connect("button_press_event", self.on_press)
        # self.viewer.dims.events.current_step.connect(self.update_plot)

        self.current_coord = None

        #events
        self.localisation_threshold.valueChanged.connect(lambda: self.update_slider_label("localisation_threshold"))
        self.localisation_area_min.valueChanged.connect(lambda: self.update_slider_label("localisation_area_min"))
        self.localisation_area_max.valueChanged.connect(lambda: self.update_slider_label("localisation_area_max"))
        self.plot_localisation_number.valueChanged.connect(lambda: self.update_slider_label("plot_localisation_number"))
        self.plot_frame_number.valueChanged.connect(lambda: self.update_slider_label("plot_frame_number"))

        self.localisation_bbox_size.valueChanged.connect(self.modify_bounding_boxes)

        self.localisation_import_image.clicked.connect(self.import_localisation_image)
        self.localisation_detect.clicked.connect(self.detect_localisations)

        self.localisation_threshold.valueChanged.connect(self.threshold_image)

        self.localisation_area_min.valueChanged.connect(self.detect_localisations)
        self.localisation_area_max.valueChanged.connect(self.detect_localisations)

        self.localisation_live_detection.stateChanged.connect(self.update_live_localisation_event)

        self.import_image.clicked.connect(self.import_image_file)

        self.load_dev.clicked.connect(self.load_dev_files)

        self.plot_mode.currentIndexChanged.connect(self.plot_graphs)
        self.plot_localisation_number.valueChanged.connect(self.plot_graphs)
        self.plot_frame_number.valueChanged.connect(self.plot_graphs)

        self.canvas.mpl_connect("button_press_event", self.update_dims_from_plot)

        self.plot_localisation_classify.clicked.connect(self.classify_localisation)
        self.plot_localisation_filter.currentIndexChanged.connect(self.filter_localisations)

        self.plot_compute.clicked.connect(self.compute_plot_data)

        self.gapseq_export_data.clicked.connect(self.export_data)

        self.gapseq_import_localisations.clicked.connect(partial(self.import_gapseq_data, mode = 'localisations'))
        self.gapseq_import_all.clicked.connect(partial(self.import_gapseq_data, mode='all'))

        self.gapseq_export_traces.clicked.connect(self.export_traces)


    def export_traces(self):

        if "bounding_boxes" in self.viewer.layers:

            if "bounding_box_data" in self.box_layer.metadata.keys():

                box_num = len(self.box_layer.data.copy())

                meta = self.box_layer.metadata.copy()

                bounding_box_data = meta["bounding_box_data"]
                bounding_box_class = meta["bounding_box_class"]

                layers = bounding_box_data.keys()

                image_trace_index = []
                image_trace_layer = []
                image_trace_class = []
                image_trace_data = []
                localisation_trace_index = []
                localisation_trace_layer = []
                localisation_trace_class = []
                localisation_trace_data = []

                for i in range(box_num):

                    for layer in layers:

                        box_data = bounding_box_data[layer][i]
                        box_class = bounding_box_class[i]

                        if layer == "localisation_image":

                            localisation_trace_data.append(box_data)

                            localisation_trace_index.append(i)
                            localisation_trace_class.append(box_class)
                            localisation_trace_layer.append(layer)

                        else:

                            image_trace_data.append(box_data)

                            image_trace_index.append(i)
                            image_trace_class.append(box_class)
                            image_trace_layer.append(layer)

                image_trace_data = pd.DataFrame(np.array(image_trace_data).T)
                localisation_trace_data = pd.DataFrame(np.array(localisation_trace_data).T)

                image_trace_data.columns = [image_trace_index,image_trace_layer,image_trace_class]
                localisation_trace_data.columns = [localisation_trace_index, localisation_trace_layer, localisation_trace_class]

                desktop = os.path.expanduser("~/Desktop")
                path = os.path.join(desktop,"gapseq_traces.xlsx")

                with pd.ExcelWriter(path) as writer:
                    localisation_trace_data.to_excel(writer, sheet_name='Localisation Data', index=True, startrow=1, startcol=1)
                    image_trace_data.to_excel(writer, sheet_name='Image Data', index=True, startrow=1,startcol=1)


    def import_gapseq_data(self, mode = "all"):

        desktop = os.path.expanduser("~/Desktop")
        path, _ = QFileDialog.getOpenFileName(self, "Open Files", desktop, "GapSeq Files (*.txt)")

        if os.path.isfile(path):

            with open(path, 'r', encoding='utf-8') as f:
                gapseq_data = json.load(f)

            bounding_boxes = gapseq_data["bounding_boxes"]

            image_layers = gapseq_data["image_layers"]
            image_paths = gapseq_data["image_paths"]
            image_metadata = gapseq_data["image_metadata"]
            localisation_threshold = gapseq_data["localisation_threshold"]
            meta = {}

            if mode == "all":

                for i in range(len(image_paths)):

                    path = image_paths[i]
                    layer_name = image_layers[i]
                    meta = image_metadata[i]

                    meta["path"] = path
                    meta["localisation_threshold"] = gapseq_data["localisation_threshold"]

                    if layer_name == "localisation_image":

                        self.import_localisation_image(meta = meta, import_gapseq=True)

                    else:

                        gap_code = meta["gap_code"]
                        seq_code = meta["seq_code"]
                        crop_mode = meta["crop_mode"]

                        layer_name = f"GAP-{gap_code}:SEQ-{seq_code}"

                        image, meta = self.read_image_file(path, crop_mode)

                        meta["gap_code"] = gap_code
                        meta["seq_code"] = seq_code

                        if layer_name in self.viewer.layers:

                            self.viewer.layers[layer_name].data = image
                            self.viewer.layers[layer_name].metadata = meta

                        else:

                            setattr(self, layer_name, self.viewer.add_image(image, name=layer_name, metadata=meta))
                            self.viewer.layers[layer_name].mouse_drag_callbacks.append(self.localisation_click_events)

            meta["bounding_box_centres"] = gapseq_data["bounding_box_centres"]
            meta["bounding_box_class"] = gapseq_data["bounding_box_class"]
            meta["localisation_threshold"] = gapseq_data["localisation_threshold"]
            meta["bounding_box_size"] = gapseq_data["bounding_box_size"]
            meta["image_layers"] = gapseq_data["image_layers"]
            meta["bounding_box_data"] = gapseq_data["bounding_box_data"]
            meta["image_paths"] = gapseq_data["image_paths"]
            meta["image_metadata"] = gapseq_data["image_metadata"]
            meta["layer_image_shape"] = gapseq_data["layer_image_shape"]

            if "bounding_boxes" in self.viewer.layers:

                self.viewer.layers["bounding_boxes"].data = bounding_boxes
                self.viewer.layers["bounding_boxes"].metadata = meta

            else:

                self.box_layer = self.viewer.add_shapes(bounding_boxes, name="bounding_boxes", shape_type='Rectangle', edge_width=1, edge_color='red', face_color=[0, 0, 0, 0], opacity=0.3, metadata=gapseq_data)
                self.box_layer.mouse_drag_callbacks.append(self.localisation_click_events)

            self.sort_layer_order()
            self.plot_graphs()


    def export_data(self):

        if "bounding_boxes" in self.viewer.layers:

            if "bounding_box_data" in self.box_layer.metadata.keys():

                bounding_boxes = self.box_layer.data.copy()

                bounding_boxes = [box.tolist() for box in bounding_boxes]

                meta = self.box_layer.metadata.copy()
#
                bounding_box_data = meta["bounding_box_data"]
                bounding_box_centres = meta["bounding_box_centres"]
                bounding_box_class = meta["bounding_box_class"]
                bounding_box_size = meta["bounding_box_size"]
                layer_image_shape = meta["layer_image_shape"]

                if "image_paths" not in meta.keys():

                    image_layers = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "localisation_threshold"]]
                    image_paths = [self.viewer.layers[layer].metadata["image_path"] for layer in image_layers]
                    image_metadata = [self.viewer.layers[layer].metadata for layer in image_layers]

                    path = self.viewer.layers["localisation_image"].metadata["path"]


                else:
                    image_paths = meta["image_paths"]
                    image_metadata = meta["image_metadata"]
                    image_layers = meta["image_layers"]

                    path = image_paths[image_layers.index("localisation_image")]

                file_name = os.path.basename(path)
                directory = path.replace(file_name,"")

                extension = file_name.split(".")[-1]
                file_name = file_name.replace("."+extension,"_gapseq.txt")

                if self.gapseq_export_at_import.isChecked() is False:

                    desktop = os.path.expanduser("~/Desktop")
                    directory = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

                if os.path.isdir(directory):

                    path = os.path.abspath(os.path.join(directory,file_name))

                    gapseq_data = dict(bounding_boxes=bounding_boxes,
                                       bounding_box_centres=bounding_box_centres,
                                       bounding_box_class=bounding_box_class,
                                       localisation_threshold = self.localisation_threshold.value(),
                                       bounding_box_size=bounding_box_size,
                                       image_layers = image_layers,
                                       bounding_box_data = bounding_box_data,
                                       image_paths = image_paths,
                                       image_metadata = image_metadata,
                                       layer_image_shape = layer_image_shape)

                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(gapseq_data, f, ensure_ascii=False, indent=4)


    def compute_plot_data(self):

        if "bounding_boxes" in self.viewer.layers:

            image_layers = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "localisation_threshold"]]

            bounding_boxes = self.box_layer.data.copy()
            meta = self.box_layer.metadata.copy()

            bounding_box_centres = meta["bounding_box_centres"]
            bounding_box_class = meta["bounding_box_class"]
            bounding_box_size = meta["bounding_box_size"]
            layer_image_shape = {}
            bounding_box_data = {}

            for i in range(len(image_layers)):

                image = self.viewer.layers[image_layers[i]].data
                layer = image_layers[i]
                bounding_box_data[layer] = []
                layer_image_shape[layer] = image.shape

                for j in range(len(bounding_boxes)):

                    progress = int(( ((i + 1) * (j +1)) / len(bounding_boxes) * len(image_layers)) * 100)

                    box = bounding_boxes[j]
                    box_centre = bounding_box_centres[j]
                    box_class = bounding_box_class[j]
                    box_size = bounding_box_size

                    [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = box

                    vertialslice = image[:, int(y1):int(y2), int(x1):int(x2)]
                    data = np.mean(vertialslice, axis=(1, 2)).tolist()

                    bounding_box_data[layer].append(data)

                    self.plot_compute_progress.setValue(progress)

            meta["bounding_box_data"] = bounding_box_data
            meta["layer_image_shape"] = layer_image_shape
            meta["image_layers"] = image_layers

            self.box_layer.metadata = meta
            self.plot_compute_progress.setValue(0)
            self.plot_graphs()


    def filter_localisations(self):

        filter_class = self.plot_localisation_filter.currentText()

        bounding_box_class = self.box_layer.metadata["bounding_box_class"]

        if filter_class != "None":

            localisation_number = bounding_box_class.count(int(filter_class))

        else:

            localisation_number = len(bounding_box_class)

        self.plot_localisation_number.setMaximum(localisation_number - 1)
        self.plot_localisation_number.setMinimum(0)

        self.plot_graphs()


    def classify_localisation(self):

        new_class = self.plot_localisation_class.currentIndex()
        localisation_number = self.plot_localisation_number.value()

        self.box_layer.metadata["bounding_box_class"][localisation_number] = int(new_class)

        self.plot_graphs()


    def load_dev_files(self):

        self.import_localisation_image()
        self.threshold_image()
        self.detect_localisations()

        path = r"C:/napari-gapseq/src/napari_gapseq/dev/20220527_27thMay2022GAP36A/27thMay2022GAPSeq4onebyonesubstrategAPGS8FRETfoursea25nM_GAPSeqonebyoneGS83TL638Exp200.tif"

        gap_codes = ["A","T","C","G"]
        seq_code = "A"
        crop_mode = self.image_import_channel.currentIndex()

        for i in range(1):

            index = len(path) - 15
            path = path[:index] + gap_codes[i] + path[index + 1:]

            gap_code = gap_codes[i]
            self.image_gap_code.setCurrentText(gap_code)

            image, meta = self.read_image_file(path, crop_mode)

            meta["gap_code"] = gap_code
            meta["seq_code"] = seq_code

            layer_name = f"GAP-{gap_code}:SEQ-{seq_code}"

            if layer_name in self.viewer.layers:

                self.viewer.layers[layer_name].data = image
                self.viewer.layers[layer_name].metadata = meta

            else:

                setattr(self, layer_name, self.viewer.add_image(image, name=layer_name, metadata=meta))
                self.viewer.layers[layer_name].mouse_drag_callbacks.append(self.localisation_click_events)

        self.sort_layer_order()
        self.compute_plot_data()


    def plot_graphs(self):

        plot_data = None
        plot_mode_index = self.plot_mode.currentIndex()

        if "bounding_boxes" in self.viewer.layers:

            if "bounding_box_data" in self.box_layer.metadata.keys():

                image_layers = self.box_layer.metadata["image_layers"]

                if plot_mode_index == 0:

                    layers = ["localisation_image"]

                    maximum_height = 300 * len(layers)
                    self.graph_container.setMaximumHeight(maximum_height)

                    bounding_box_data = self.box_layer.metadata["bounding_box_data"]
                    layer_image_shape = self.box_layer.metadata["layer_image_shape"]

                    frame_num = np.max([layer_image_shape[layer][0] for layer in layers])

                    self.plot_frame_number.setMaximum(frame_num-1)
                    current_step = list(self.viewer.dims.current_step)
                    current_step[0] = self.plot_frame_number.value()
                    self.viewer.dims.current_step = tuple(current_step)

                    plot_data = self.get_plot_data(layers=["localisation_image"])
                    self.plot(plot_data=plot_data)

                if plot_mode_index == 1:

                    layers = [layer for layer in image_layers if layer not in ["localisation_image", "localisation_threshold", "bounding_boxes"]]

                    maximum_height = 300 * len(layers)
                    self.graph_container.setMaximumHeight(maximum_height)

                    if len(layers) > 0:

                        layer_image_shape = self.box_layer.metadata["layer_image_shape"]

                        frame_num = np.max([layer_image_shape[layer][0] for layer in layers])
                        self.plot_frame_number.setMaximum(frame_num - 1)
                        current_step = list(self.viewer.dims.current_step)
                        current_step[0] = self.plot_frame_number.value()
                        self.viewer.dims.current_step = tuple(current_step)

                        layers = sorted(layers)

                        plot_data = self.get_plot_data(layers=layers)
                        self.plot(plot_data=plot_data)

                if plot_mode_index == 2:

                    layers = [layer for layer in image_layers if layer not in ["bounding_boxes", "localisation_threshold"]]

                    maximum_height = 300 * len(layers)
                    self.graph_container.setMaximumHeight(maximum_height)

                    if len(layers) > 0:

                        layers = sorted(layers)
                        layers.insert(0,layers.pop(layers.index("localisation_image")))

                        layer_image_shape = self.box_layer.metadata["layer_image_shape"]

                        frame_num = np.max([layer_image_shape[layer][0] for layer in layers])

                        self.plot_frame_number.setMaximum(frame_num - 1)
                        current_step = list(self.viewer.dims.current_step)
                        current_step[0] = self.plot_frame_number.value()
                        self.viewer.dims.current_step = tuple(current_step)

                        plot_data = self.get_plot_data(layers=layers)
                        self.plot(plot_data=plot_data)

                if self.plot_localisation_focus.isChecked() and plot_data is not None:

                    x1,x2,y1,y2 = plot_data[0]["box"]
                    image_shape = plot_data[0]["image_shape"]

                    centre = (0, y1 + (y2 - y1) // 2, x1 + (x2 - x1) // 2)
                    zoom = min((image_shape[0]/(y2-y1)), (image_shape[1]/(x2-x1)))/2

                    self.viewer.camera.center = centre
                    self.viewer.camera.zoom = zoom


    def update_dims_from_plot(self, event):

        if event.xdata != None:

            frame_int = int(event.xdata)
            ax = event.inaxes

            lines = ax.get_lines()
            layer = [str(line.get_label()) for line in lines if str(line.get_label()) != "Frame"][0]

            self.plot_frame_number.setValue(frame_int)

            if layer in self.viewer.layers:

                layer_index = self.viewer.layers.index(layer)
                num_layers = len(self.viewer.layers) - 1

                self.viewer.layers.move(layer_index, num_layers)


    def get_plot_data(self, layers):

        filter_class = self.plot_localisation_filter.currentText()

        if "bounding_box_data" in self.box_layer.metadata.keys():

            meta = self.box_layer.metadata.copy()

            bounding_box_data = meta["bounding_box_data"]

            plot_data = []

            for layer in layers:

                bounding_box_layer_data = bounding_box_data[layer]
                image_shape = meta["layer_image_shape"][layer]

                if filter_class == "None":

                    frame_number = self.plot_frame_number.value()
                    localisation_number = self.plot_localisation_number.value()

                    if localisation_number != -1:

                        bounding_box = self.box_layer.data[localisation_number]
                        bounding_box_class = self.box_layer.metadata["bounding_box_class"][localisation_number]

                        data = bounding_box_layer_data[localisation_number]

                        [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = bounding_box

                        if frame_number > len(data):
                            frame_number = len(data) - 1

                        plot_data.append({'layer_name': layer, 'data': data, "image_shape":image_shape,
                                          "current_frame":frame_number, "box": [x1,x2,y1,y2],
                                          "bounding_box_class": bounding_box_class,
                                          "localisation_number": localisation_number})

                if filter_class != "None":

                    frame_number = self.plot_frame_number.value()
                    localisation_number = self.plot_localisation_number.value()

                    bounding_box_class = self.box_layer.metadata["bounding_box_class"]
                    localisation_positions = np.where(np.array(bounding_box_class) == int(filter_class))[0].tolist()

                    if len(localisation_positions) > 0:

                        localisation_number = localisation_positions[localisation_number]

                        if localisation_number != -1:

                            self.label.setText(str(localisation_number))

                            bounding_box = self.box_layer.data[localisation_number]
                            bounding_box_class = self.box_layer.metadata["bounding_box_class"][localisation_number]

                            if int(filter_class) == bounding_box_class and localisation_number >= 0:

                                data = bounding_box_layer_data[localisation_number]

                                [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = bounding_box

                                if frame_number > len(data):
                                    frame_number = len(data) - 1

                                plot_data.append({'layer_name': layer, 'data': data, "image_shape":image_shape,
                                                  "current_frame": frame_number, "box": [x1, x2, y1, y2],
                                                  "bounding_box_class": bounding_box_class,
                                                  "localisation_number": localisation_number})

        return plot_data


    def plot(self, plot_data, xlabel = "Frame", ylabel = "Intensity"):

        if len(plot_data) > 0:

            self.canvas.figure.clf()

            bounding_box_class = plot_data[0]["bounding_box_class"]
            self.plot_localisation_class.setCurrentIndex(bounding_box_class)

            for i in range(len(plot_data)):

                plot_index = int(f"{len(plot_data)}1{i+1}")

                axes = self.canvas.figure.add_subplot(plot_index)
                layer_name = plot_data[i]["layer_name"]
                bounding_box_class = plot_data[i]["bounding_box_class"]
                localisation_number = plot_data[i]["localisation_number"]

                y = plot_data[i]["data"]
                x = np.arange(len(y))

                axes.set_facecolor("#262930")
                axes.plot(x, y, label = layer_name)
                axes.plot([], [], ' ', label=f"#:{localisation_number}   Class:{bounding_box_class}")
                axes.legend(loc="upper right")

                ymin, ymax = axes.get_ylim()
                current_frame = int(self.viewer.dims.current_step[0])

                if current_frame < len(x):
                    axes.plot([current_frame,current_frame],[ymin, ymax], color = "red", label = "Frame")

            self.canvas.figure.tight_layout()
            self.canvas.draw()

        else:
            self.canvas.figure.clf()
            self.canvas.draw()


    def update_live_localisation_event(self):

        if self.localisation_live_detection.isChecked():

            self.localisation_area_min.valueChanged.connect(self.detect_localisations)
            self.localisation_area_max.valueChanged.connect(self.detect_localisations)

        else:

            self.localisation_area_min.valueChanged.disconnect(self.detect_localisations)
            self.localisation_area_max.valueChanged.disconnect(self.detect_localisations)



    def localisation_click_events(self, viewer, event):

        if "Control" in event.modifiers and "bounding_boxes" in self.viewer.layers:

            selected_layer = self.viewer.layers.selection.active
            data_coordinates = selected_layer.world_to_data(event.position)
            coord = np.round(data_coordinates).astype(int)
            self.current_coord = coord

            if coord is not None:

                if len(coord) > 2:
                    coord = coord[1:]

                bounding_boxes = self.box_layer.data
                meta = self.box_layer.metadata.copy()

                bounding_box_centres = meta["bounding_box_centres"]

                box_index = self.box_layer.get_value(coord)[0]

                if box_index is not None:

                    del bounding_boxes[box_index]
                    del bounding_box_centres[box_index]

                    meta["bounding_box_centres"] = bounding_box_centres

                    self.box_layer.data = bounding_boxes
                    self.box_layer.metadata = meta

                else:

                    size = self.localisation_bbox_size.value()

                    box = [[int(coord[0] - size), int(coord[1] - size)], [int(coord[0] + size), int(coord[1] + size)]]
                    box_centre = [int(coord[0]), int(coord[1])]

                    bounding_boxes.append(box)
                    bounding_box_centres.append(box_centre)

                    meta["bounding_box_size"] = size
                    meta["bounding_box_centres"] = bounding_box_centres

                    self.box_layer.data = bounding_boxes
                    self.box_layer.metadata = meta


    def modify_bounding_boxes(self):

        if "bounding_boxes" in self.viewer.layers:

            bounding_boxes = self.box_layer.data.copy()
            meta = self.box_layer.metadata.copy()

            bounding_box_centres = meta["bounding_box_centres"]

            size = self.localisation_bbox_size.value()

            for i in range(len(bounding_boxes)):

                cx,cy = bounding_box_centres[i]

                box = [[cy - size, cx - size], [cy + size, cx + size]]

                bounding_boxes[i] = box

            if len(bounding_boxes) > 0 :

                meta["bounding_box_size"] = size

                self.box_layer.data = bounding_boxes
                self.box_layer.metadata = meta


    def threshold_image(self):

        if "localisation_threshold" in self.viewer.layers:

            threshold_image = self.localisation_threshold_layer.data
            meta = self.localisation_threshold_layer.metadata

            localisation_mask_image = threshold_image[:,: threshold_image.shape[1]//2]
            localisation_mask = threshold_image[:, threshold_image.shape[1]//2 : ]

            localisation_threshold = self.localisation_threshold.value()

            _, localisation_mask = cv.threshold(localisation_mask_image, localisation_threshold, 255, cv.THRESH_BINARY)

            localisation_threshold_image = np.hstack((localisation_mask_image, localisation_mask))

            meta["localisation_threshold"] = localisation_threshold

            self.localisation_threshold_layer.data = localisation_threshold_image
            self.localisation_threshold_layer.metadata = meta


    def detect_localisations(self):

        if "localisation_threshold" in self.viewer.layers:

            localisation_area_min_value = self.localisation_area_min.value()
            localisation_area_max_value = self.localisation_area_max.value()
            bounding_box_size = self.localisation_bbox_size.value()

            threshold_image = self.localisation_threshold_layer.data
            meta = self.localisation_threshold_layer.metadata

            localisation_mask = threshold_image[:,threshold_image.shape[1]//2 :]

            contours = find_contours(localisation_mask)

            polygons = []
            bounding_box_centres = []
            bounding_box_class = []

            for i in range(len(contours)):

                try:

                    cnt = contours[i]

                    x, y, w, h = cv.boundingRect(cnt)

                    area = cv2.contourArea(cnt)

                    if area > localisation_area_min_value and area < localisation_area_max_value:

                        M = cv.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        size = bounding_box_size

                        polygon = [[cy - size, cx - size], [cy + size, cx + size]]

                        polygons.append(polygon)
                        bounding_box_centres.append([cx,cy])
                        bounding_box_class.append(0)

                except:
                    pass

            if len(polygons) > 0:

                meta["bounding_box_centres"] = bounding_box_centres
                meta["bounding_box_class"] = bounding_box_class
                meta["bounding_box_size"] = bounding_box_size

                self.plot_localisation_number.setMaximum(len(polygons)-1)

                if "bounding_boxes" in self.viewer.layers:

                    self.box_layer.data = polygons
                    self.box_layer.metadata = meta

                else:
                    self.box_layer = self.viewer.add_shapes(polygons, name = "bounding_boxes", shape_type='Rectangle', edge_width=1, edge_color='red', face_color=[0, 0, 0, 0], opacity=0.3, metadata=meta)
                    self.box_layer.mouse_drag_callbacks.append(self.localisation_click_events)

    def import_localisation_image(self, meta = [], import_gapseq = False):

        path = r"C:/napari-gapseq/src/napari_gapseq/dev/20220527_27thMay2022GAP36A/27thMay2022GAPSeq4onebyonesubstrategAPGS8FRETfoursea50nMconce2_GAPSeqonebyoneGAP36AL532Exp200.tif"
        # path = r"C:/napari-gapseq/src/napari_gapseq/dev/20220527_27thMay2022GAP36A/27thMay2022GAPSeq4onebyonesubstrategAPGS8FRETfoursea50nMconce2_GAPSeqonebyoneGAP36AL532Exp200_locImage.tiff"


        if import_gapseq is True:

            localisation_threshold = meta["localisation_threshold"]
            crop_mode = meta["crop_mode"]
            path = meta["path"]

            self.localisation_threshold.setValue(localisation_threshold)

        else:

            localisation_threshold = self.localisation_threshold.value()
            crop_mode = self.localisation_channel.currentIndex()

            desktop = os.path.expanduser("~/Desktop")
            path, _ = QFileDialog.getOpenFileName(self, "Open Files", desktop, "NIM Image File (*.tif);; NIM Localisation File (*.tiff)")

        if os.path.isfile(path):

            if "_locImage" in path:

                image, meta = self.read_image_file(path, 0)
                meta["crop_mode"] = 0

                image = np.moveaxis(image, -1, 0)
                image = np.mean(image, axis=0).astype(np.uint8)

                _, localisation_mask = cv.threshold(image, localisation_threshold, 255, cv.THRESH_BINARY)

                localisation_threshold_image = np.hstack((image,localisation_mask))

                meta["localisation_threshold"] = localisation_threshold
                meta["path"] = path

            else:

                image, meta = self.read_image_file(path,crop_mode)
                meta["crop_mode"] = crop_mode

                img = np.max(image, axis=0)
                img = normalize99(img)
                img = rescale01(img) * 255
                img = img.astype(np.uint8)

                localisation_mask_image = cv.fastNlMeansDenoising(img, h=30, templateWindowSize=5, searchWindowSize=31)
                _, localisation_mask = cv.threshold(localisation_mask_image, localisation_threshold, 255, cv.THRESH_BINARY)

                localisation_threshold_image = np.hstack((localisation_mask_image,localisation_mask))

                meta["localisation_threshold"] = localisation_threshold
                meta["path"] = path


            if "localisation_image" in self.viewer.layers:

                self.localisation_image_layer.data = image
                self.localisation_image_layer.metadata = meta
                self.localisation_threshold_layer.data = localisation_threshold_image
                self.localisation_threshold_layer.metadata = meta

            else:

                self.localisation_image_layer = self.viewer.add_image(image, name="localisation_image",metadata=meta)
                self.localisation_threshold_layer = self.viewer.add_image(localisation_threshold_image, name="localisation_threshold", metadata=meta)

                self.localisation_image_layer.mouse_drag_callbacks.append(self.localisation_click_events)
                self.localisation_threshold_layer.mouse_drag_callbacks.append(self.localisation_click_events)

            self.plot_frame_number.setMaximum(image.shape[0]-1)

            self.viewer.reset_view()
            self.sort_layer_order()


    def import_image_file(self):

        crop_mode = self.image_import_channel.currentIndex()
        gap_code = self.image_gap_code.currentText()
        seq_code = self.image_sequence_code.currentText()

        desktop = os.path.expanduser("~/Desktop")
        directory = r"C:\napari-gapseq\src\napari_gapseq\dev\20220527_27thMay2022GAP36A"
        path, filter = QFileDialog.getOpenFileName(self, "Open Files", directory, "Files (*.tif)")

        if os.path.isfile(path):

            image, meta = self.read_image_file(path, crop_mode)

            meta["gap_code"] = gap_code
            meta["seq_code"] = seq_code

            layer_name = f"GAP-{gap_code}:SEQ-{seq_code}"

            if layer_name in self.viewer.layers:

                self.viewer.layers[layer_name].data = image
                self.viewer.layers[layer_name].metadata = meta

            else:

                setattr(self, layer_name, self.viewer.add_image(image, name=layer_name, metadata=meta))
                self.viewer.layers[layer_name].mouse_drag_callbacks.append(self.localisation_click_events)

            self.sort_layer_order()


    def sort_layer_order(self):

        image_layers = self.viewer.layers

        if "bounding_boxes" in image_layers:

            layer_index = self.viewer.layers.index("bounding_boxes")
            num_layers = len(self.viewer.layers)

            self.viewer.layers.move(layer_index, num_layers)

    def read_image_file(self, path, crop_mode=0):

        image_name = os.path.basename(path)

        with tifffile.TiffFile(path) as tif:
            try:
                metadata = tif.pages[0].tags["ImageDescription"].value
                metadata = json.loads(metadata)
            except:
                metadata = {}

            image = tifffile.imread(path)

        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=0)

        image = self.crop_image(image, crop_mode)

        folder = os.path.abspath(path).split("\\")[-2]
        parent_folder = os.path.abspath(path).split("\\")[-3]

        if "image_name" not in metadata.keys():

            metadata["image_name"] = image_name
            metadata["channel"] = None
            metadata["segmentation_file"] = None
            metadata["segmentation_channel"] = None
            metadata["image_path"] = path
            metadata["mask_name"] = None
            metadata["mask_path"] = None
            metadata["label_name"] = None
            metadata["label_path"] = None
            metadata["crop_mode"] = crop_mode
            metadata["folder"] = folder
            metadata["parent_folder"] = parent_folder
            metadata["image_shape"] = image.shape
            metadata["dims"] = [image.shape[-1], image.shape[-2]]
            metadata["crop"] = [0, image.shape[-2], 0, image.shape[-1]]

        return image, metadata

    def crop_image(self, img, crop_mode=0):

        if crop_mode != 0:

            if len(img.shape) > 2:
                imgL = img[:, :, :img.shape[-1] // 2]
                imgR = img[:, :, img.shape[-1] // 2:]
            else:
                imgL = img[:, :img.shape[-1] // 2]
                imgR = img[:, img.shape[-1] // 2:]

            if crop_mode == 1:
                img = imgL
            if crop_mode == 2:
                img = imgR

            if crop_mode == 3:
                if np.mean(imgL) > np.mean(imgR):
                    img = imgL
                else:
                    img = imgR

        return img

    def update_slider_label(self, slider_name):

        label_name = slider_name + "_label"

        self.slider = self.findChild(QSlider, slider_name)
        self.label = self.findChild(QLabel, label_name)

        if slider_name == "plot_localisation_number" and self.plot_localisation_filter.currentText() != "None":

            pass

        else:

            slider_value = self.slider.value()
            self.label.setText(str(slider_value))



    def on_press(self, event):

        if event.xdata != None:

            frame_int = int(event.xdata)

            current_dims = self.viewer.dims.current_step

            new_dims = tuple([frame_int, *list(current_dims)[1:]])

            self.viewer.dims.current_step = new_dims


    # def import_localisation_file(self):
    #
    #     # desktop = os.path.expanduser("~/Desktop")
    #     # path, filter = QFileDialog.getOpenFileName(self, "Open Files", desktop, "Files (*)")
    #
    #     path = r"C:/napari-gapseq/src/napari_gapseq/dev/20220527_27thMay2022GAP36A/27thMay2022GAPSeq4onebyonesubstrategAPGS8FRETfoursea50nMconce2_GAPSeqonebyoneGAP36AL532Exp200.tif"
    #
    #     image = tifffile.imread(path)
    #
    #     image = image[:, :, : image.shape[2] // 2]
    #
    #     img = np.max(image, axis=0)
    #
    #     img = normalize99(img)
    #     img = rescale01(img) * 255
    #     img = img.astype(np.uint8)
    #
    #     img = cv.fastNlMeansDenoising(img, h=30, templateWindowSize=5, searchWindowSize=31)
    #
    #     _, mask = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
    #
    #     contours = find_contours(mask)
    #
    #     polygons = []
    #
    #     for i in range(len(contours)):
    #
    #         try:
    #
    #             cnt = contours[i]
    #
    #             x, y, w, h = cv.boundingRect(cnt)
    #
    #             M = cv.moments(cnt)
    #             cx = int(M['m10'] / M['m00'])
    #             cy = int(M['m01'] / M['m00'])
    #
    #             polygon = np.array([[cy - 2, cx - 2], [cy + 2, cx + 2]])
    #
    #             polygons.append(polygon)
    #
    #             # cv.rectangle(img, (cx - 2, cy - 2), (cx + 2, cy + 2), (0, 128, 255), 2)
    #
    #         except:
    #             pass
    #
    #
    #     self.localisation_layer = self.viewer.add_image(image, name="Localisations")
    #     self.localisation_layer.mouse_drag_callbacks.append(partial(self.select_localisation,mode="click"))
    #
    #     self.box_layer = self.viewer.add_shapes(polygons, shape_type='Rectangle', edge_width=1, edge_color='red', face_color=[0,0,0,0], opacity=0.3)
    #     self.box_layer.mouse_drag_callbacks.append(partial(self.select_localisation,mode="click"))
    #
    #     self.viewer.layers.selection.events.changed.connect(partial(self.select_localisation,mode="layer_change"))

    # def import_image_file(self):
    #
    #     path = r"C:/napari-gapseq/src/napari_gapseq/dev/20220527_27thMay2022GAP36A/27thMay2022GAPSeq4onebyonesubstrategAPGS8FRETfoursea25nM_GAPSeqonebyoneGS83TL638Exp200.tif"
    #
    #     gap_codes = ["A","T","C","G"]
    #     layer_names = []
    #     for i in range(len(gap_codes)):
    #
    #         index = len(path) - 15
    #         path = path[:index] + gap_codes[i] + path[index + 1:]
    #
    #         image = tifffile.imread(path)
    #
    #         image = image[:, :, image.shape[2] // 2 : ]
    #
    #         layer_name = f"SEQ-{gap_codes[i]}"
    #         layer_names.append(layer_name)
    #
    #         setattr(self, layer_name, self.viewer.add_image(image, name=layer_name))
    #         self.viewer.layers[layer_name].mouse_drag_callbacks.append(partial(self.select_localisation, mode="click"))


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


    # def plot(self, canvas, plot_data, xlabel = "Frame", ylabel = "Intensity"):
    #
    #     canvas.figure.clf()
    #
    #     maximum_width = 200*len(plot_data)
    #
    #     if maximum_width != 0:
    #         self.graph_container.setMaximumHeight(maximum_width)
    #
    #     for i in range(len(plot_data)):
    #
    #         plot_index = int(f"{len(plot_data)}1{i+1}")
    #         # spot_index = int(f"{len(plot_data)}2{i+2}")
    #
    #         axes = canvas.figure.add_subplot(plot_index)
    #         layer_name = plot_data[i]["layer_name"]
    #         spot = plot_data[i]["spot"]
    #
    #         y = plot_data[i]["data"]
    #         x = np.arange(len(y))
    #
    #         axes.set_facecolor("#262930")
    #         axes.plot(x, y, label = layer_name)
    #         axes.legend()
    #
    #         ymin, ymax = axes.get_ylim()
    #         current_frame = int(self.viewer.dims.current_step[0])
    #
    #         axes.plot([current_frame,current_frame],[ymin, ymax], color = "red", label = "Frame")
    #
    #         # spot_axes = canvas.figure.add_subplot(spot_index)
    #         # spot_axes.imshow(spot)
    #
    #     self.canvas.figure.tight_layout()
    #     canvas.draw()

    # def update_plot(self):
    #
    #     allaxes = self.canvas.figure.get_axes()
    #
    #     if len(allaxes) > 0:
    #
    #         current_frame = int(self.viewer.dims.current_step[0])
    #
    #         for ax in allaxes:
    #
    #             # print(ax.get_gridspec())
    #
    #             vertical_line = ax.lines[1]
    #             vertical_line.set_xdata([current_frame, current_frame])
    #             self.canvas.draw()
    #             self.canvas.flush_events()
    #
    #
