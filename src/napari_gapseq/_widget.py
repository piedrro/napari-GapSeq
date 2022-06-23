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
from skimage.filters import difference_of_gaussians
from skimage.morphology import erosion, disk

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
from scipy import optimize
import warnings
from scipy.spatial import distance

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

    def __init__(self, viewer: napari.Viewer):

        super().__init__()

        from napari_gapseq.gapseq_ui import Ui_TabWidget

        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.form = Ui_TabWidget()
        self.akseg_ui = QTabWidget()
        self.form.setupUi(self.akseg_ui)
        self.layout().addWidget(self.akseg_ui)

        #register QWidgets/Controls
        self.localisation_channel = self.findChild(QComboBox,"localisation_channel")
        self.localisation_import_image = self.findChild(QPushButton,"localisation_import_image")
        self.localisation_type = self.findChild(QComboBox,"localisation_type")
        self.localisation_threshold = self.findChild(QSlider,"localisation_threshold")
        self.localisation_area_min = self.findChild(QSlider,"localisation_area_min")
        self.localisation_area_max = self.findChild(QSlider,"localisation_area_max")
        self.localisation_bbox_size = self.findChild(QSlider,"localisation_bbox_size")
        self.localisation_threshold_label = self.findChild(QLabel,"localisation_threshold_label")
        self.localisation_area_min_label = self.findChild(QLabel,"localisation_area_min_label")
        self.localisation_area_max_label = self.findChild(QLabel,"localisation_area_max_label")
        self.localisation_aspect_ratio = self.findChild(QSlider,"localisation_aspect_ratio")
        self.localisation_minimum_distance = self.findChild(QSlider,"localisation_minimum_distance")
        self.localisation_bbox_size_label = self.findChild(QLabel,"localisation_bbox_size_label")
        self.localisation_detect = self.findChild(QPushButton,"localisation_detect")
        self.load_dev = self.findChild(QPushButton,"load_dev")
        self.plot_compute = self.findChild(QPushButton,"plot_compute")
        self.plot_compute_progress = self.findChild(QProgressBar, "plot_compute_progress")
        self.plot_mode = self.findChild(QComboBox,"plot_mode")
        self.plot_localisation_number = self.findChild(QSlider,"plot_localisation_number")
        self.plot_localisation_number_label = self.findChild(QLabel,"plot_localisation_number_label")
        self.plot_frame_number = self.findChild(QSlider,"plot_frame_number")

        self.plot_nucleotide_class = self.findChild(QComboBox,"plot_nucleotide_class")
        self.plot_nucleotide_classify = self.findChild(QPushButton,"plot_nucleotide_classify")
        self.plot_localisation_class = self.findChild(QComboBox,"plot_localisation_class")
        self.plot_localisation_classify = self.findChild(QPushButton,"plot_localisation_classify")
        self.plot_localisation_filter = self.findChild(QComboBox,"plot_localisation_filter")
        self.plot_nucleotide_filter = self.findChild(QComboBox, "plot_nucleotide_filter")

        self.plot_localisation_focus = self.findChild(QCheckBox,"plot_localisation_focus")
        self.plot_show_active = self.findChild(QCheckBox,"plot_show_active")
        self.plot_background_subtraction_mode = self.findChild(QComboBox,"plot_background_subtraction_mode")
        self.graph_container = self.findChild(QWidget,"graph_container")
        self.gapseq_export_data = self.findChild(QPushButton,"gapseq_export_data")
        self.gapseq_export_at_import = self.findChild(QCheckBox,"gapseq_export_at_import")
        self.gapseq_import_localisations = self.findChild(QPushButton,"gapseq_import_localisations")
        self.gapseq_import_all = self.findChild(QPushButton, "gapseq_import_all")
        self.gapseq_export_traces = self.findChild(QPushButton, "gapseq_export_traces")
        self.gapseq_export_traces_filter = self.findChild(QComboBox, "gapseq_export_traces_filter")

        self.graph_container.setLayout(QVBoxLayout())
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

        self.current_coord = None

        #events
        self.localisation_threshold.valueChanged.connect(lambda: self.update_slider_label("localisation_threshold"))
        self.localisation_area_min.valueChanged.connect(lambda: self.update_slider_label("localisation_area_min"))
        self.localisation_area_max.valueChanged.connect(lambda: self.update_slider_label("localisation_area_max"))
        self.localisation_aspect_ratio.valueChanged.connect(lambda: self.update_slider_label("localisation_aspect_ratio"))
        self.localisation_bbox_size.valueChanged.connect(lambda: self.update_slider_label("localisation_bbox_size"))
        self.localisation_minimum_distance.valueChanged.connect(lambda: self.update_slider_label("localisation_minimum_distance"))
        self.plot_localisation_number.valueChanged.connect(lambda: self.update_slider_label("plot_localisation_number"))
        self.plot_frame_number.valueChanged.connect(lambda: self.update_slider_label("plot_frame_number"))

        self.update_slider_label("localisation_aspect_ratio")

        self.localisation_bbox_size.valueChanged.connect(self.modify_bounding_boxes)

        self.localisation_import_image.clicked.connect(self.import_localisation_image)
        self.localisation_detect.clicked.connect(self.detect_localisations)

        self.localisation_threshold.valueChanged.connect(self.threshold_image)

        self.import_image.clicked.connect(self.import_image_file)

        self.load_dev.clicked.connect(self.load_dev_files)

        self.plot_mode.currentIndexChanged.connect(self.plot_graphs)
        self.plot_localisation_number.valueChanged.connect(self.plot_graphs)
        self.plot_frame_number.valueChanged.connect(self.plot_graphs)

        self.canvas.mpl_connect("button_press_event", self.update_dims_from_plot)

        self.plot_localisation_classify.clicked.connect(self.classify_localisation)
        self.plot_nucleotide_classify.clicked.connect(self.classify_nucleotide)

        self.plot_localisation_filter.currentIndexChanged.connect(self.filter_localisations)
        self.plot_nucleotide_filter.currentIndexChanged.connect(self.filter_localisations)

        self.plot_background_subtraction_mode.currentIndexChanged.connect(self.plot_graphs)

        self.plot_compute.clicked.connect(self.compute_plot_data)

        self.gapseq_export_data.clicked.connect(self.export_data)

        self.gapseq_import_localisations.clicked.connect(partial(self.import_gapseq_data, mode = 'localisations'))
        self.gapseq_import_all.clicked.connect(partial(self.import_gapseq_data, mode='all'))
        self.gapseq_export_traces.clicked.connect(self.export_traces)

        self.viewer.bind_key(key="Alt-1", func=partial(self.keybind_classify_class, key=0), overwrite=True)
        self.viewer.bind_key(key="Alt-1", func=partial(self.keybind_classify_class, key=1), overwrite=True)
        self.viewer.bind_key(key="Alt-2", func=partial(self.keybind_classify_class, key=2), overwrite=True)
        self.viewer.bind_key(key="Alt-3", func=partial(self.keybind_classify_class, key=3), overwrite=True)
        self.viewer.bind_key(key="Alt-4", func=partial(self.keybind_classify_class, key=4), overwrite=True)
        self.viewer.bind_key(key="Alt-5", func=partial(self.keybind_classify_class, key=5), overwrite=True)
        self.viewer.bind_key(key="Alt-6", func=partial(self.keybind_classify_class, key=6), overwrite=True)
        self.viewer.bind_key(key="Alt-7", func=partial(self.keybind_classify_class, key=7), overwrite=True)
        self.viewer.bind_key(key="Alt-8", func=partial(self.keybind_classify_class, key=8), overwrite=True)
        self.viewer.bind_key(key="Alt-9", func=partial(self.keybind_classify_class, key=9), overwrite=True)

        self.viewer.bind_key(key="Alt-a", func=partial(self.keybind_classify_nucleotide, key="A"), overwrite=True)
        self.viewer.bind_key(key="Alt-c", func=partial(self.keybind_classify_nucleotide, key="C"), overwrite=True)
        self.viewer.bind_key(key="Alt-t", func=partial(self.keybind_classify_nucleotide, key="T"), overwrite=True)
        self.viewer.bind_key(key="Alt-g", func=partial(self.keybind_classify_nucleotide, key="G"), overwrite=True)

        self.viewer.bind_key(key="d", func=partial(self.keybind_delete_event, key='d'), overwrite=True)



    def keybind_delete_event(self,viewer,key):

        if "bounding_boxes" in self.viewer.layers:

            if "bounding_box_data" in self.box_layer.metadata.keys():

                localisation_number = self.plot_localisation_number.value()

                localisation_number = int(self.plot_localisation_number_label.text())

                bounding_boxes = self.box_layer.data.copy()
                meta = self.box_layer.metadata.copy()

                bounding_box_centres = meta["bounding_box_centres"]
                bounding_box_class = meta["bounding_box_class"]
                nucleotide_class = meta["nucleotide_class"]
                bounding_box_data = meta["bounding_box_data"]
                background_data = meta["background_data"]

                if localisation_number is not None:

                    del bounding_boxes[localisation_number]
                    del bounding_box_centres[localisation_number]
                    del bounding_box_class[localisation_number]
                    del nucleotide_class[localisation_number]

                    for layer in bounding_box_data.keys():

                        del bounding_box_data[layer][localisation_number]
                        del background_data[layer]["local_background"][localisation_number]

                    meta["bounding_box_centres"] = bounding_box_centres
                    meta["bounding_box_class"] = bounding_box_class
                    meta["bounding_box_data"] = bounding_box_data
                    meta["background_data"] = background_data
                    meta["nucleotide_class"] = nucleotide_class

                    self.box_layer.data = bounding_boxes
                    self.box_layer.metadata = meta

                    self.plot_localisation_number.setMaximum(len(bounding_boxes) - 1)

                    self.plot_graphs()

    def keybind_classify_nucleotide(self, viewer, key):

        if "bounding_boxes" in self.viewer.layers:

            if "bounding_box_data" in self.box_layer.metadata.keys():

                localisation_number = self.plot_localisation_number.value()

                self.box_layer.metadata["nucleotide_class"][localisation_number] = str(key)

                self.plot_graphs()

    def keybind_classify_class(self, viewer, key):

        if "bounding_boxes" in self.viewer.layers:

            if "bounding_box_data" in self.box_layer.metadata.keys():

                localisation_number = self.plot_localisation_number.value()

                self.box_layer.metadata["bounding_box_class"][localisation_number] = int(key)

                self.plot_graphs()

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
            meta["background_data"] = gapseq_data["background_data"]
            meta["nucleotide_class"] = gapseq_data["nucleotide_class"]
            meta["localisation_type"] = gapseq_data["localisation_type"]

            if "bounding_boxes" in self.viewer.layers:

                if meta["localisation_type"] == "Box":
                    self.box_layer.shape_type = ["Rectangle"] * len(bounding_boxes)

                if meta["localisation_type"] == "Circle":
                    self.box_layer.shape_type = ["Ellipse"] * len(bounding_boxes)

                self.viewer.layers["bounding_boxes"].data = bounding_boxes
                self.viewer.layers["bounding_boxes"].metadata = meta

            else:

                if meta["localisation_type"] == "Box":

                    self.box_layer = self.viewer.add_shapes(bounding_boxes, name="bounding_boxes", shape_type='Rectangle', edge_width=1, edge_color='red', face_color=[0, 0, 0, 0], opacity=0.3, metadata=gapseq_data)
                    self.box_layer.mouse_drag_callbacks.append(self.localisation_click_events)

                if meta["localisation_type"] == "Circle":

                    self.box_layer = self.viewer.add_shapes(bounding_boxes, name="bounding_boxes", shape_type='Ellipse',edge_width=1, edge_color='red', face_color=[0, 0, 0, 0],opacity=0.3,metadata=meta)
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
                background_data = meta["background_data"]
                nucleotide_class = meta["nucleotide_class"]
                localisation_type = meta["localisation_type"]


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
                                       layer_image_shape = layer_image_shape,
                                       background_data = background_data,
                                       nucleotide_class = nucleotide_class,
                                       localisation_type = localisation_type)

                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(gapseq_data, f, ensure_ascii=False, indent=4)

    def get_background_mask(self, bounding_boxes,bounding_box_size, bounding_box_centres, image):

        bounding_boxes = self.box_layer.data.copy()
        shape_type = np.unique(self.box_layer.shape_type)[0]

        background_mask = np.zeros((image.shape[-2], image.shape[-1]), dtype=np.uint8)

        for i in range(len(bounding_boxes)):
            polygon = bounding_boxes[i]
            [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = polygon
            cx, cy = bounding_box_centres[i]
            box_size = bounding_box_size

            if shape_type == "rectangle":
                background_mask[int(y1):int(y2), int(x1):int(x2)] = 255
            if shape_type == "ellipse":
                cv.circle(background_mask, (cx, cy), box_size, 255, -1)

        background_image = image.copy()
        masked_image = image.copy()

        background_image[:,background_mask == 255] = 0
        masked_image[:, background_mask != 255] = 0

        return background_image, masked_image

    def compute_plot_data(self):

        if "bounding_boxes" in self.viewer.layers:

            image_layers = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "localisation_threshold"]]

            bounding_boxes = self.box_layer.data.copy()

            shape_type = np.unique(self.box_layer.shape_type)[0]
            meta = self.box_layer.metadata.copy()

            bounding_box_centres = meta["bounding_box_centres"]
            bounding_box_class = meta["bounding_box_class"]
            bounding_box_size = meta["bounding_box_size"]
            layer_image_shape = {}
            bounding_box_data = {}
            background_data = {}

            for i in range(len(image_layers)):

                image = self.viewer.layers[image_layers[i]].data
                layer = image_layers[i]
                bounding_box_data[layer] = []
                layer_image_shape[layer] = image.shape

                background_image, masked_image = self.get_background_mask(bounding_boxes, bounding_box_size, bounding_box_centres, image)

                background_data[layer]= {"global_background": np.mean(background_image,axis=(1, 2)).tolist(),
                                         "local_background": []}

                for j in range(len(bounding_boxes)):

                    progress = int(( ((i + 1) * (j +1)) / len(bounding_boxes) * len(image_layers)) * 100)

                    polygon = bounding_boxes[j]
                    [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = polygon
                    cx,cy = bounding_box_centres[j]
                    box_class = bounding_box_class[j]
                    box_size = bounding_box_size
                    background_box_size = 10

                    data = masked_image[:, int(y1):int(y2), int(x1):int(x2)]
                    data = np.nanmean(data, axis=(1, 2)).tolist()

                    [[y1,x1],[y2,x2]] = [[cy - background_box_size, cx - background_box_size],
                                         [cy + background_box_size, cx + background_box_size]]
                    local_background_data = background_image[:, int(y1):int(y2), int(x1):int(x2)].copy()
                    local_background_data = np.mean(local_background_data, axis=(1, 2)).tolist()

                    bounding_box_data[layer].append(data)
                    background_data[layer]["local_background"].append(local_background_data)

                    self.plot_compute_progress.setValue(progress)

            meta["bounding_box_data"] = bounding_box_data
            meta["layer_image_shape"] = layer_image_shape
            meta["image_layers"] = image_layers
            meta["background_data"] = background_data

            self.box_layer.metadata = meta
            self.plot_compute_progress.setValue(0)
            self.plot_graphs()

    def filter_localisations(self):

        localisation_filter = self.plot_localisation_filter.currentText()
        nucleotide_filter = self.plot_nucleotide_filter.currentText()

        bounding_box_class = self.box_layer.metadata["bounding_box_class"]
        nucleotide_class = self.box_layer.metadata["nucleotide_class"]

        if localisation_filter != "None":
            box_class_localisations = np.where(np.array(bounding_box_class) == int(localisation_filter))[0].tolist()
        else:
            box_class_localisations = np.arange(len(self.box_layer.data))

        if nucleotide_filter != "None":
            nucleotide_class_localisations = np.where(np.array(nucleotide_class) == nucleotide_filter)[0].tolist()
        else:
            nucleotide_class_localisations = np.arange(len(self.box_layer.data))

        localisation_positions = list(set(box_class_localisations).intersection(nucleotide_class_localisations))

        self.plot_localisation_number.setMaximum(len(localisation_positions)-1)
        self.plot_localisation_number.setMinimum(0)

        self.plot_graphs()

    def classify_localisation(self):

        new_class = self.plot_localisation_class.currentIndex()
        localisation_number = self.plot_localisation_number.value()

        self.box_layer.metadata["bounding_box_class"][localisation_number] = int(new_class)

        self.plot_graphs()

    def classify_nucleotide(self):

        new_class = self.plot_nucleotide_class.currentText()
        localisation_number = self.plot_localisation_number.value()

        self.box_layer.metadata["nucleotide_class"][localisation_number] = str(new_class)

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


                if self.plot_show_active.isChecked():

                    edge_colours = [[1.0,0.0,0.0,0.0] for i in self.box_layer.data]

                    localisation_number = int(self.plot_localisation_number_label.text())

                    edge_colours[localisation_number] = [1.0,0.0,0.0,1.0]

                    self.box_layer.edge_color = edge_colours

                else:
                    edge_colours = [[1.0,0.0,0.0,1.0] for i in self.box_layer.data]
                    self.box_layer.edge_color = edge_colours

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

        localisation_filter = self.plot_localisation_filter.currentText()
        nucleotide_filter = self.plot_nucleotide_filter.currentText()

        if self.plot_localisation_filter.currentText() != "None" or self.plot_nucleotide_filter.currentText() != "None":

            filter_plots = True

        if "bounding_box_data" in self.box_layer.metadata.keys():

            meta = self.box_layer.metadata.copy()

            bounding_box_data = meta["bounding_box_data"]

            if "background_data" in meta.keys():
                background_data = meta["background_data"]
            else:
                background_data = None

            plot_background_subtraction_mode = self.plot_background_subtraction_mode.currentIndex()

            plot_data = []

            for layer in layers:

                bounding_box_layer_data = bounding_box_data[layer]
                image_shape = meta["layer_image_shape"][layer]

                if localisation_filter == "None" and nucleotide_filter == "None":

                    self.plot_localisation_number.setMaximum(len(self.box_layer.data) - 1)

                    frame_number = self.plot_frame_number.value()
                    localisation_number = self.plot_localisation_number.value()

                    if localisation_number != -1:

                        bounding_box = self.box_layer.data[localisation_number]
                        bounding_box_class = self.box_layer.metadata["bounding_box_class"][localisation_number]
                        nucleotide_class = self.box_layer.metadata["nucleotide_class"][localisation_number]

                        data = bounding_box_layer_data[localisation_number]

                        if plot_background_subtraction_mode == 1 and background_data != None:

                            background = background_data[layer]["local_background"][localisation_number]
                            data = list(np.array(data) - np.array(background))
                            data = list(data - np.min(data))

                        if plot_background_subtraction_mode == 2 and background_data != None:

                            background = background_data[layer]["global_background"]
                            data = np.array(data) - np.array(background)
                            data = list(data - np.min(data))

                        [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = bounding_box

                        if frame_number > len(data):
                            frame_number = len(data) - 1

                        plot_data.append({'layer_name': layer, 'data': data, "image_shape":image_shape,
                                          "current_frame":frame_number, "box": [x1,x2,y1,y2],
                                          "bounding_box_class": bounding_box_class,
                                          "localisation_number": localisation_number,
                                          "nucleotide_class":nucleotide_class})

                else:

                    frame_number = self.plot_frame_number.value()
                    localisation_number = self.plot_localisation_number.value()

                    bounding_box_class = self.box_layer.metadata["bounding_box_class"]
                    nucleotide_class = self.box_layer.metadata["nucleotide_class"]

                    if localisation_filter != "None":
                        box_class_localisations = np.where(np.array(bounding_box_class) == int(localisation_filter))[0].tolist()
                    else:
                        box_class_localisations = np.arange(len(self.box_layer.data))

                    if nucleotide_filter != "None":
                        nucleotide_class_localisations = np.where(np.array(nucleotide_class) == nucleotide_filter)[0].tolist()
                    else:
                        nucleotide_class_localisations = np.arange(len(self.box_layer.data))

                    localisation_positions = list(set(box_class_localisations).intersection(nucleotide_class_localisations))

                    if len(localisation_positions) > 0:

                        localisation_number = localisation_positions[localisation_number]

                        if localisation_number != -1:

                            self.label.setText(str(localisation_number))

                            bounding_box = self.box_layer.data[localisation_number]
                            bounding_box_class = self.box_layer.metadata["bounding_box_class"][localisation_number]
                            nucleotide_class = self.box_layer.metadata["nucleotide_class"][localisation_number]

                            if localisation_number >= 0:

                                data = bounding_box_layer_data[localisation_number]

                                if plot_background_subtraction_mode == 1 and background_data != None:
                                    background = background_data[layer]["local_background"][localisation_number]
                                    data = list(np.array(data) - np.array(background))
                                    data = list(data - np.min(data))

                                if plot_background_subtraction_mode == 2 and background_data != None:
                                    background = background_data[layer]["global_background"]
                                    data = np.array(data) - np.array(background)
                                    data = list(data - np.min(data))

                                [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = bounding_box

                                if frame_number > len(data):
                                    frame_number = len(data) - 1

                                plot_data.append({'layer_name': layer, 'data': data, "image_shape":image_shape,
                                                  "current_frame": frame_number, "box": [x1, x2, y1, y2],
                                                  "bounding_box_class": bounding_box_class,
                                                  "localisation_number": localisation_number,
                                                  "nucleotide_class":nucleotide_class})

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
                nucleotide_class = plot_data[i]["nucleotide_class"]

                y = plot_data[i]["data"]
                x = np.arange(len(y))

                axes.set_facecolor("#262930")
                axes.plot(x, y, label = layer_name)
                axes.plot([], [], ' ', label=f"#:{localisation_number}  C:{bounding_box_class}  N:{nucleotide_class}")
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

    def localisation_click_events(self, viewer, event):

        if "Control" in event.modifiers and "bounding_boxes" in self.viewer.layers:

            selected_layer = self.viewer.layers.selection.active
            data_coordinates = selected_layer.world_to_data(event.position)
            coord = np.round(data_coordinates).astype(int)
            self.current_coord = coord
            shape_type = np.unique(self.box_layer.shape_type)[0]

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

                    cx,cy = int(coord[1]), int(coord[0])

                    polygon = [[cy - size, cx - size],
                               [cy + size, cx - size],
                               [cy + size, cx + size],
                               [cy - size, cx + size]]

                    box_centre = [cx, cy]

                    bounding_boxes.append(polygon)
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

                polygon = [[cy - size, cx - size],
                           [cy + size, cx - size],
                           [cy + size, cx + size],
                           [cy - size, cx + size]]

                bounding_boxes[i] = polygon

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

    def filter_array(self, array, condition):

        array = [elem for i, elem in enumerate(array) if condition[i] == True]

        return array

    def detect_localisations(self):

        if "localisation_threshold" in self.viewer.layers:

            localisation_area_min_value = self.localisation_area_min.value()
            localisation_area_max_value = self.localisation_area_max.value()
            bounding_box_size = self.localisation_bbox_size.value()

            threshold_image = self.localisation_threshold_layer.data

            meta = self.localisation_threshold_layer.metadata

            localisation_mask = threshold_image[:,threshold_image.shape[1]//2 :]
            localisation_image = threshold_image[:,: threshold_image.shape[1]//2]

            contours = find_contours(localisation_mask)

            bounding_boxes = []
            bounding_circles = []
            bounding_box_centres = []
            bounding_box_class = []
            nucleotide_class = []

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

                        polygon = [[cy - size, cx - size],
                                   [cy + size, cx - size],
                                   [cy + size, cx + size],
                                   [cy - size, cx + size]]

                        bounding_boxes.append(polygon)
                        bounding_box_centres.append([cx,cy])
                        bounding_box_class.append(0)
                        nucleotide_class.append("N/A")

                except:
                    pass

            distances = distance.cdist(bounding_box_centres, bounding_box_centres)

            distances[distances == 0] = np.nan
            distances = np.nanmin(distances,axis=1)

            distance_filter = distances > self.localisation_minimum_distance.value()

            bounding_boxes = self.filter_array(bounding_boxes, distance_filter)
            bounding_box_class = self.filter_array(bounding_box_class, distance_filter)
            nucleotide_class = self.filter_array(nucleotide_class, distance_filter)
            bounding_box_centres = self.filter_array(bounding_box_centres, distance_filter)

            if len(bounding_boxes) > 0:

                meta["bounding_box_centres"] = bounding_box_centres
                meta["bounding_box_class"] = bounding_box_class
                meta["bounding_box_size"] = bounding_box_size
                meta["nucleotide_class"] = nucleotide_class
                meta["localisation_type"] = self.localisation_type.currentText()

                self.plot_localisation_number.setMaximum(len(bounding_boxes)-1)

                if "bounding_boxes" in self.viewer.layers:

                    if self.localisation_type.currentText() == "Box":
                        self.box_layer.shape_type = ["Rectangle"]*len(bounding_boxes)

                    if self.localisation_type.currentText() == "Circle":
                        self.box_layer.shape_type = ["Ellipse"]*len(bounding_boxes)

                    self.box_layer.data = bounding_boxes
                    self.box_layer.metadata = meta

                else:

                    if self.localisation_type.currentText() == "Box":

                        self.box_layer = self.viewer.add_shapes(bounding_boxes, name = "bounding_boxes", shape_type='Rectangle', edge_width=1, edge_color='red', face_color=[0, 0, 0, 0], opacity=0.3, metadata=meta)

                    if self.localisation_type.currentText() == "Circle":

                        self.box_layer = self.viewer.add_shapes(bounding_boxes, name="bounding_boxes", shape_type='Ellipse',edge_width=1, edge_color='red', face_color=[0, 0, 0, 0],opacity=0.3, metadata=meta)


                self.fit_localisations()
                self.box_layer.mouse_drag_callbacks.append(self.localisation_click_events)

    def fit_localisations(self):

        threshold_image = self.localisation_threshold_layer.data
        localisation_image = threshold_image[:, : threshold_image.shape[1] // 2]
        localisation_image = np.expand_dims(localisation_image,0)

        aspect_ratio_max = int(self.localisation_aspect_ratio.value())/10

        bounding_boxes = self.box_layer.data
        meta = self.box_layer.metadata.copy()

        bounding_box_size = meta["bounding_box_size"]
        bounding_box_centres = meta["bounding_box_centres"]

        background_image, masked_image = self.get_background_mask(bounding_boxes, bounding_box_size, bounding_box_centres, localisation_image)

        fitted_bounding_boxes = []
        fitted_bounding_box_centres = []
        fitted_bounding_box_class = []
        fitted_nucleotide_class = []

        for i in range(len(bounding_boxes)):

            polygon = bounding_boxes[i]

            cx, cy = bounding_box_centres[i]
            size = bounding_box_size

            box = [[cy - size, cx - size], [cy + size, cx + size]]
            [[y1, x1], [y2, x2]] = box

            img = masked_image[0][y1:y2, x1:x2]

            try:

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    params = self.fitgaussian(img)

                cy = int(cy + 0.5 - size + params[1])
                cx = int(cx + 0.5 - size + params[2])

                aspect_ratio = np.max([params[3], params[4]]) / np.min([params[3], params[4]])

                if aspect_ratio < aspect_ratio_max:

                    polygon = [[cy - size, cx - size],
                               [cy + size, cx - size],
                               [cy + size, cx + size],
                               [cy - size, cx + size]]

                    fitted_bounding_boxes.append(polygon)
                    fitted_bounding_box_centres.append([cx, cy])
                    fitted_bounding_box_class.append(0)
                    fitted_nucleotide_class.append("N/A")


            except:
                pass

        meta["bounding_box_centres"] = fitted_bounding_box_centres
        meta["bounding_box_class"] = fitted_bounding_box_class
        meta["nucleotide_class"] = fitted_nucleotide_class

        self.box_layer.data = fitted_bounding_boxes
        self.box_layer.metadata = meta

    def gaussian(self, height, center_x, center_y, width_x, width_y):

        """Returns a gaussian function with the given parameters"""

        width_x = float(width_x)
        width_y = float(width_y)

        return lambda x, y: height * np.exp(
            -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)

    def moments(self, data):

        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """

        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X * data).sum() / total
        y = (Y * data).sum() / total

        col = data[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size) - x) ** 2 * col).sum() / col.sum())

        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size) - y) ** 2 * row).sum() / row.sum())

        height = data.max()

        return height, x, y, width_x, width_y

    def fitgaussian(self, data):

        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""

        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)

        return p

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

                image = difference_of_gaussians(image, 1)

                _, localisation_mask = cv.threshold(image, localisation_threshold, 255, cv.THRESH_BINARY)

                footprint = disk(1)
                image = erosion(image, footprint)

                localisation_threshold_image = np.hstack((image,localisation_mask))

                meta["localisation_threshold"] = localisation_threshold
                meta["path"] = path

            else:

                image, meta = self.read_image_file(path,crop_mode)
                meta["crop_mode"] = crop_mode

                img = np.max(image, axis=0)

                img = difference_of_gaussians(img, 1)

                img = normalize99(img)
                img = rescale01(img) * 255
                img = img.astype(np.uint8)

                # localisation_mask_image = cv.fastNlMeansDenoising(img, h=30, templateWindowSize=5, searchWindowSize=31)

                footprint = disk(1)
                localisation_mask_image = erosion(img, footprint)

                _, localisation_mask = cv.threshold(img, localisation_threshold, 255, cv.THRESH_BINARY)

                localisation_threshold_image = np.hstack((img,localisation_mask))

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

            if slider_name == "localisation_aspect_ratio":

                slider_value = self.slider.value()/10

            self.label.setText(str(slider_value))
