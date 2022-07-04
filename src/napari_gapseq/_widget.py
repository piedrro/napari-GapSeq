"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

from qtpy.QtWidgets import (QWidget,QVBoxLayout,QTabWidget,QCheckBox,QLabel,QLineEdit,QFileDialog,
                            QComboBox,QPushButton,QProgressBar,QTextEdit,QSlider,QSpinBox, QSpacerItem, QSizePolicy)
from PyQt5 import QtWidgets, QtCore
from qtpy.QtCore import (QObject,QRunnable,QThreadPool,)
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
from matplotlib.widgets import Button
from matplotlib.backends.backend_qt5agg import FigureCanvasQT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import time
import json
from scipy import optimize
import warnings
from scipy.spatial import distance
import ruptures as rpt
import scipy
plt.style.use('dark_background')

from typing import TYPE_CHECKING
from multiprocessing import Pool, shared_memory

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari




class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

    def result(self):

        return self.fn(*self.args, **self.kwargs)






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




def gaussian(height, center_x, center_y, width):

    """Returns a gaussian function with the given parameters"""

    width = float(width)

    return lambda x, y: height * np.exp(
        -(((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2) / 2)

def moments(data):

    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """

    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total

    col = data[:, int(y)]
    width = np.sqrt(np.abs((np.arange(col.size) - x) ** 2 * col).sum() / col.sum())

    height = data.max()

    return height, x, y, width

def fitgaussian(data, params = []):

    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""

    if len(params) == 0:
        params = moments(data)

    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)

    return p, success


def crop_image(img, crop_mode=0):

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

def stack_image(image, stack_mode):

    if stack_mode == 0:

        image = np.mean(image,axis=0)

    if stack_mode == 1:

        image = np.max(image,axis=0)

    if stack_mode == 2:

        image = np.std(image,axis=0)

    return image

def compute_undrift_localisations(image, box_centres, loc0_data=None, box_size=5):

    if loc0_data != None:
        loc0_centers = loc0_data["loc_centers"]
        param_list = loc0_data["loc_params"]
    else:
        loc0_centers = box_centres
        param_list = [[]] * len(box_centres)

    loc_bboxs = []
    loc_centers = []
    loc_params = []

    for i in range(len(loc0_centers)):

        try:

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                loc = loc0_centers[i]
                params = param_list[i]

                cx, cy = loc

                x1, x2 = cx - box_size, cx + box_size
                y1, y2 = cy - box_size, cy + box_size

                img = image[int(y1):int(y2), int(x1):int(x2)]

                params, _ = fitgaussian(img, params)

                cy = cy + 1 - box_size + params[1]
                cx = cx + 1 - box_size + params[2]

                x1, x2 = cx - box_size, cx + box_size
                y1, y2 = cy - box_size, cy + box_size

                loc_bboxs.append([x1, x2, y1, y2])
                loc_centers.append([cx, cy])
                loc_params.append(params)

        except:
            pass

    return loc_bboxs,loc_centers,loc_params


def load_undrift_segment(index_list, path, crop_mode,stack_mode, box_centres, loc0_data=None):

    with tifffile.TiffFile(path) as tif:

        img = [page.asarray() for index, page in enumerate(tif.pages) if index in index_list]

        img = np.array(img)
        img = crop_image(img,crop_mode)
        img = stack_image(img,stack_mode)

        loc_bboxs, loc_centers, loc_params = compute_undrift_localisations(img, box_centres, loc0_data)

        loc_data = dict(index_list = index_list, loc_bboxs=loc_bboxs,loc_centers=loc_centers,loc_params=loc_params)

    return loc_data


def process_loc_data(loc_data, n_frames):

    loc0_centres = np.array(loc_data.pop(0)["loc_centers"])

    drift = [[0, 0]]
    index = [0]

    for i in range(len(loc_data)):

        loc_centres = np.array(loc_data[i]["loc_centers"])

        distances = distance.cdist(loc0_centres, loc_centres)

        distances[distances == 0] = np.nan
        distances[distances > 3] = np.nan

        x_drift = []
        y_drift = []

        for j in range(distances.shape[0]):

            dat = distances[j]

            try:
                loc_index = np.nanargmin(dat)
                loc0_index = j

                loc_centre = loc_centres[loc_index]
                loc0_centre = loc0_centres[loc0_index]

                difference = loc0_centre - loc_centre

                x_drift.append(loc0_centre[0] - loc_centre[0])
                y_drift.append(loc0_centre[1] - loc_centre[1])
                index.append()

            except:
                pass

        drift.append([np.mean(x_drift), np.mean(y_drift)])
        index.append(np.array(loc_data[i]["index_list"])[0])

    shift_y = np.array(drift)[:, 0]
    shift_x = np.array(drift)[:, 1]

    shift_y = np.interp(np.arange(n_frames), index, shift_y)
    shift_x = np.interp(np.arange(n_frames), index, shift_x)

    drift = np.stack((shift_x, shift_y)).T

    return drift


def undrift_image(index, path, drift):

    with tifffile.TiffFile(path) as tif:
        img = np.array(tif.pages[index].asarray())

        img = scipy.ndimage.shift(img, drift[index])

        return [index, img]


def threshold_localisation_image(image, threshold, box_min, box_max):

    img = image

    img = difference_of_gaussians(img, 1)

    img = normalize99(img)
    img = rescale01(img) * 255

    img = img.astype(np.uint8)

    _, mask = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

    footprint = disk(1)
    mask = erosion(mask, footprint)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > box_min and cv.contourArea(cnt) < box_max]

    mask = np.zeros_like(img, dtype=np.uint8)
    mask = cv.drawContours(mask, contours, -1, 255, -1)

    return mask


def compute_box_stats(box_data, shared_image_object, shared_background_object, threshold, box_min, box_max):

    try:

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            frame = box_data[0]["frame"]

            image_shm = shared_memory.SharedMemory(
                name=shared_image_object.name)
            image = np.ndarray(
                box_data[0]["image_shape"], dtype=box_data[0]["image_type"], buffer=image_shm.buf)[frame]

            background_shm = shared_memory.SharedMemory(
                name=shared_background_object.name)
            background_image = np.ndarray(
                box_data[0]["image_shape"], dtype=box_data[0]["image_type"], buffer=background_shm.buf)[frame]

            localisation_mask = threshold_localisation_image(image, threshold, box_min, box_max)

            for i in range(len(box_data)):
                box = box_data[i]["box"]
                [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = box

                cx, cy = box_data[i]["box_centre"]
                box_class = box_data[i]["box_class"]
                box_size = box_data[i]["box_size"]

                background_box_size = 10
                img = image[int(y1):int(y2), int(x1):int(x2)].copy()
                mask = localisation_mask[int(y1):int(y2), int(x1):int(x2)].copy()

                [[y1,x1],[y2,x2]] = [[cy - background_box_size, cx - background_box_size],
                                      [cy + background_box_size, cx + background_box_size]]

                local_background_data = background_image[int(y1):int(y2), int(x1):int(x2)].copy()

                box_data[i]["box_mean"] = np.nanmean(img)
                box_data[i]["box_mean_global_background"] = np.nanmean(background_image)
                box_data[i]["box_mean_local_background"] = np.nanmean(local_background_data)

                box_data[i]["box_std"] = np.nanstd(img)
                box_data[i]["box_std_global_background"] = np.nanstd(background_image)
                box_data[i]["box_std_local_background"] = np.nanstd(local_background_data)

                box_data[i]["box_range"] = np.nanmax(img) - np.nanmin(img)
                box_data[i]["box_range_global_background"] = np.nanmax(background_image) - np.nanmin(background_image)
                box_data[i]["box_range_local_background"] = np.nanmax(local_background_data) - np.nanmin(local_background_data)

                params, success = fitgaussian(img)

                gaussian_x = cx + 1 - box_size + params[2]
                gaussian_y = cy + 1 - box_size + params[1]

                fit_error = False
                if gaussian_x < x1 or gaussian_x > x2:
                    fit_error = True
                if gaussian_y < y1 or gaussian_y > y2:
                    fit_error = True
                if success != 1:
                    fit_error = True
                if np.max(mask) == 0:
                    fit_error = True

                if fit_error is True:
                    params = [0,0,0,0,0]
                    gaussian_x = cy + 1 - box_size + params[1]
                    gaussian_y = cx + 1 - box_size + params[2]

                box_data[i]["box_class"] = box_class
                box_data[i]["gaussian_height"] = params[0]
                box_data[i]["gaussian_x"] = gaussian_x
                box_data[i]["gaussian_y"] = gaussian_y
                box_data[i]["gausian_width"] = params[3]

            del image
            del background_image
            image_shm.close()
            background_shm.close()


    except:
        import traceback
        print(traceback.format_exc())
        pass

    return box_data

class json_np_encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

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
        self.localisation_stack_mode = self.findChild(QComboBox,"localisation_stack_mode")
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
        self.localisation_undrift = self.findChild(QPushButton,"localisation_undrift")
        self.undrift_progress = self.findChild(QProgressBar,"undrift_progress")
        self.localisation_frame_averaging = self.findChild(QComboBox,"localisation_frame_averaging")
        self.plot_compute = self.findChild(QPushButton,"plot_compute")
        self.plot_compute_progress = self.findChild(QProgressBar, "plot_compute_progress")
        self.plot_mode = self.findChild(QComboBox,"plot_mode")
        self.plot_metric = self.findChild(QComboBox,"plot_metric")
        self.plot_localisation_number = self.findChild(QSlider,"plot_localisation_number")
        self.plot_localisation_number_label = self.findChild(QLabel,"plot_localisation_number_label")
        self.plot_frame_number = self.findChild(QSlider,"plot_frame_number")
        self.plot_normalise = self.findChild(QCheckBox, "plot_normalise")
        self.fit_plot_normalise = self.findChild(QCheckBox, "fit_plot_normalise")

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
        self.export_traces_at_import = self.findChild(QCheckBox, "export_traces_at_import")
        self.gapseq_export_traces_excel = self.findChild(QPushButton, "gapseq_export_traces_excel")
        self.gapseq_export_traces_dat = self.findChild(QPushButton, "gapseq_export_traces_dat")
        self.traces_class_filter = self.findChild(QComboBox, "traces_class_filter")
        self.traces_nucleotide_filter = self.findChild(QComboBox,"traces_nucleotide_filter")
        self.traces_data_selection = self.findChild(QComboBox,"traces_data_selection")
        self.traces_background_mode = self.findChild(QComboBox, "traces_background_mode")
        self.traces_data_metric = self.findChild(QComboBox,"traces_data_metric")
        self.fit_traces_progress = self.findChild(QProgressBar,"fit_traces_progress")
        self.export_progress = self.findChild(QProgressBar,"export_progress")

        self.image_import_channel = self.findChild(QComboBox,"image_import_channel")
        self.image_gap_code = self.findChild(QComboBox,"image_gap_code")
        self.image_sequence_code = self.findChild(QComboBox, "image_sequence_code")
        self.import_image = self.findChild(QPushButton,"import_image")

        self.import_localisations = self.findChild(QPushButton, "import_localisations")
        self.import_image = self.findChild(QPushButton, "import_image")

        self.current_coord = None

        self.fit_localisation_number = self.findChild(QSlider,"fit_localisation_number")
        self.fit_graph_container = self.findChild(QWidget, "fit_graph_container")
        self.fit_plot_channel = self.findChild(QComboBox, "fit_plot_channel")
        self.fit_plot_metric = self.findChild(QComboBox, "fit_plot_metric")
        self.fit_cpd_mode = self.findChild(QComboBox,"fit_cpd_mode")
        self.fit_active = self.findChild(QPushButton,"fit_active")
        self.fit_all = self.findChild(QPushButton,"fit_all")
        self.fit_background_subtraction_mode = self.findChild(QComboBox,"fit_background_subtraction_mode")

        self.fit_cpd_model = self.findChild(QComboBox,"fit_cpd_model")
        self.fit_cpd_penalty = self.findChild(QSpinBox,"fit_cpd_penalty")
        self.fit_cpd_window_size = self.findChild(QSpinBox, "fit_cpd_window_size")
        self.fit_cpd_min_size = self.findChild(QSpinBox, "fit_cpd_min_size")
        self.fit_cpd_breakpoints = self.findChild(QSpinBox,"fit_cpd_breakpoints")
        self.fit_cpd_model_label = self.findChild(QLabel,"fit_cpd_model_label")
        self.fit_cpd_penalty_label = self.findChild(QLabel,"fit_cpd_penalty_label")
        self.fit_cpd_window_size_label = self.findChild(QLabel, "fit_cpd_window_size_label")
        self.fit_cpd_min_size_label = self.findChild(QLabel, "fit_cpd_min_size_label")
        self.fit_cpd_breakpoints_label = self.findChild(QLabel,"fit_cpd_breakpoints_label")
        self.fit_cpd_jump = self.findChild(QSpinBox,"fit_cpd_jump")
        self.fit_cpd_jump_label = self.findChild(QLabel,"fit_cpd_jump_label")
        self.export_cpd_data = self.findChild(QCheckBox,"export_cpd_data")
        self.show_cpd_breakpoints = self.findChild(QCheckBox,"show_cpd_breakpoints")
        self.show_cpd_states = self.findChild(QCheckBox,"show_cpd_states")
        self.fit_hmm_states = self.findChild(QSpinBox,"fit_hmm_states")

        #create matplotib plot graph
        self.graph_container.setLayout(QVBoxLayout())
        self.graph_container.setMinimumWidth(100)
        self.canvas = FigureCanvasQTAgg()
        self.canvas.figure.set_tight_layout(True)
        self.canvas.figure.patch.set_facecolor("#262930")
        self.graph_container.layout().addWidget(self.canvas)

        #create matplotlib fit graph
        self.fit_graph_container.setLayout(QVBoxLayout())
        self.fit_graph_container.setMinimumWidth(100)
        self.fit_graph_canvas = FigureCanvasQTAgg()
        self.fit_graph_canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.fit_graph_canvas.setFocus()
        self.fit_graph_canvas.figure.set_tight_layout(True)
        self.fit_graph_canvas.figure.patch.set_facecolor("#262930")
        self.fit_graph_container.layout().addWidget(self.fit_graph_canvas)

        #events
        self.localisation_threshold.valueChanged.connect(lambda: self.update_slider_label("localisation_threshold"))
        self.localisation_area_min.valueChanged.connect(lambda: self.update_slider_label("localisation_area_min"))
        self.localisation_area_max.valueChanged.connect(lambda: self.update_slider_label("localisation_area_max"))
        self.localisation_aspect_ratio.valueChanged.connect(lambda: self.update_slider_label("localisation_aspect_ratio"))
        self.localisation_bbox_size.valueChanged.connect(lambda: self.update_slider_label("localisation_bbox_size"))
        self.localisation_minimum_distance.valueChanged.connect(lambda: self.update_slider_label("localisation_minimum_distance"))
        self.plot_localisation_number.valueChanged.connect(lambda: self.update_slider_label("plot_localisation_number"))
        self.plot_frame_number.valueChanged.connect(lambda: self.update_slider_label("plot_frame_number"))
        self.fit_localisation_number.valueChanged.connect(lambda: self.update_slider_label("fit_localisation_number"))

        self.fit_cpd_mode.currentIndexChanged.connect(self.update_cpd_controls)
        self.show_cpd_breakpoints.stateChanged.connect(self.plot_fit_graph)
        self.fit_plot_metric.currentIndexChanged.connect(self.plot_fit_graph)
        self.show_cpd_states.stateChanged.connect(self.plot_fit_graph)

        self.update_slider_label("localisation_aspect_ratio")

        self.localisation_bbox_size.valueChanged.connect(self.modify_bounding_boxes)

        self.localisation_import_image.clicked.connect(self.import_localisation_image)
        self.localisation_detect.clicked.connect(self.detect_localisations)

        self.localisation_threshold.valueChanged.connect(self.threshold_image)

        self.import_image.clicked.connect(self.import_image_file)

        self.plot_mode.currentIndexChanged.connect(self.plot_graphs)
        self.plot_metric.currentIndexChanged.connect(self.plot_graphs)
        self.plot_localisation_number.valueChanged.connect(self.plot_graphs)
        self.plot_frame_number.valueChanged.connect(self.plot_graphs)

        self.canvas.mpl_connect("button_press_event", self.update_dims_from_plot)
        self.fit_graph_canvas.mpl_connect("button_press_event", self.manual_break_point_edit)
        self.fit_graph_canvas.mpl_connect('scroll_event', self.fit_graph_zoom)
        self.fit_graph_canvas.mpl_connect('key_press_event', self.manual_state_edit)


        self.plot_localisation_classify.clicked.connect(self.classify_localisation)
        self.plot_nucleotide_classify.clicked.connect(self.classify_nucleotide)

        self.plot_localisation_filter.currentIndexChanged.connect(self.filter_localisations)
        self.plot_nucleotide_filter.currentIndexChanged.connect(self.filter_localisations)

        self.plot_background_subtraction_mode.currentIndexChanged.connect(self.plot_graphs)

        self.plot_compute.clicked.connect(self.initalise_gapseq_compute_traces)

        self.gapseq_export_data.clicked.connect(self.export_data)

        self.localisation_undrift.clicked.connect(self.initialise_gapseq_undrift)

        self.gapseq_import_localisations.clicked.connect(partial(self.import_gapseq_data, mode = 'localisations'))
        self.gapseq_import_all.clicked.connect(partial(self.import_gapseq_data, mode='all'))

        self.gapseq_export_traces_excel.clicked.connect(partial(self.threaded_export_traces, mode = "excel"))
        self.gapseq_export_traces_dat.clicked.connect(partial(self.threaded_export_traces, mode="dat"))

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

        self.fit_active.clicked.connect(partial(self.change_point_detection_mp, detection_mode="active"))
        self.fit_all.clicked.connect(partial(self.change_point_detection_mp, detection_mode="all"))


        self.fit_localisation_number.valueChanged.connect(self.plot_fit_graph)
        self.fit_plot_channel.currentIndexChanged.connect(self.plot_fit_graph)
        self.fit_background_subtraction_mode.currentIndexChanged.connect(self.plot_fit_graph)
        self.update_cpd_controls()

        self.threadpool = QThreadPool()

        # self.import_gapseq_data(mode="all",path=r"C:/napari-gapseq/src/napari_gapseq/dev/devdata.txt")
        # self.change_point_detection()

        # self.import_localisation_image()
        # self.detect_localisations()

    def get_gapseq_compute_iterator(self, layer):

        meta = self.box_layer.metadata.copy()

        bounding_box_centres = meta["bounding_box_centres"]
        bounding_box_class = meta["bounding_box_class"]
        bounding_box_size = meta["bounding_box_size"]

        bounding_boxes = self.box_layer.data.copy()

        image = self.viewer.layers[layer].data

        compute_iterator = []

        for frame in range(image.shape[0]):

            for i in range(10):

                box = bounding_boxes[i].tolist()
                box_centre = bounding_box_centres[i]
                box_class = bounding_box_class[i]

                mp_dict = dict(frame=frame, box=box, box_centre=box_centre, box_class=box_class,
                               box_size=bounding_box_size, image_shape=image.shape, image_type=image.dtype,
                               box_index=i, layer=layer)

                compute_iterator.append(mp_dict)

        compute_iterator = np.array_split(compute_iterator, image.shape[0])

        return compute_iterator


    def process_localisation_data(self, localisation_data):

        data = []

        for i in range(len(localisation_data)):

            for j in range(len(localisation_data[0])):
                data.append(localisation_data[i][j])

        localisation_data = pd.DataFrame(data)

        localisation_data = localisation_data.groupby(["box_index"])

        localisation_dict = {}

        layer = localisation_data.get_group(list(localisation_data.groups)[0])["layer"].unique()[0]

        localisation_dict[layer] = {}

        for i in range(len(localisation_data)):
            data = localisation_data.get_group(list(localisation_data.groups)[i])

            data = data.sort_values(["frame", "box_index"])

            localisation_dict[layer][i] = {"box_mean": data.box_mean.tolist(),
                                           "box_mean_local_background": data.box_mean_local_background.tolist(),
                                           "box_mean_global_background": data.box_mean_global_background.tolist(),
                                           "box_std": data.box_std.tolist(),
                                           "box_std_local_background": data.box_std_local_background.tolist(),
                                           "box_std_global_background": data.box_std_global_background.tolist(),
                                           "box_range": data.box_range.tolist(),
                                           "box_range_local_background": data.box_range_local_background.tolist(),
                                           "box_range_global_background": data.box_range_global_background.tolist(),
                                           "gaussian_height": data.gaussian_height.tolist(),
                                           "gaussian_x": data.gaussian_x.tolist(),
                                           "gaussian_y": data.gaussian_y.tolist(),
                                           "gausian_width": data.gausian_width.tolist(),
                                           "image_shape": data.image_shape.tolist()[0],
                                           "bounding_box": data.box.tolist()[0]}

        return localisation_dict



    def gapseq_compute_traces(self, progress_callback, layer):

        bounding_boxes = self.box_layer.data.copy()
        meta = self.box_layer.metadata.copy()

        bounding_box_centres = meta["bounding_box_centres"]
        bounding_box_class = meta["bounding_box_class"]
        bounding_box_size = meta["bounding_box_size"]

        image = self.viewer.layers[layer].data
        background_image, masked_image = self.get_background_mask(bounding_boxes, bounding_box_size, bounding_box_centres, image)

        shared_image_object = shared_memory.SharedMemory(create=True, size=image.nbytes)
        shared_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shared_image_object.buf)
        shared_image[:] = image[:]

        shared_background_object = shared_memory.SharedMemory(create=True, size=background_image.nbytes)
        shared_background = np.ndarray(background_image.shape, dtype=background_image.dtype, buffer=shared_background_object.buf)
        shared_background[:] = background_image[:]

        box_min = self.localisation_area_min.value()
        box_max = self.localisation_area_max.value()
        threshold = self.localisation_threshold.value()

        compute_iterator = self.get_gapseq_compute_iterator(layer)

        with Pool() as p:

            def callback(*args):
                iter.append(1)
                progress = (len(iter) / len(compute_iterator)) * 100

                if progress_callback != None:
                    progress_callback.emit(progress)

                return

            iter = []

            results = [p.apply_async(compute_box_stats, args=(i,), kwds={'shared_image_object': shared_image_object,
                                                                          'shared_background_object': shared_background_object,
                                                                          'threshold':threshold,
                                                                          'box_min':box_min,
                                                                          'box_max':box_max}, callback=callback) for i in compute_iterator]

            localisation_data = [r.get() for r in results]

            localisation_data = self.process_localisation_data(localisation_data)

            return localisation_data


    def process_gapseq_compute_traces(self, localisation_data):

        meta = self.box_layer.metadata.copy()

        if "localisation_data" in meta.keys():

            for layer,data in localisation_data.items():

                meta["localisation_data"][layer] = data

        else:
            meta["localisation_data"] = localisation_data

        layer_image_shape = {}
        bounding_box_breakpoints = {}
        bounding_box_traces = {}

        image_layers = [layer.name for layer in self.viewer.layers if
                        layer.name not in ["bounding_boxes", "localisation_threshold"]]

        for layer in image_layers:

            image = self.viewer.layers[layer].data
            data = localisation_data[layer][0]["box_mean"]
            layer_image_shape[layer] = image.shape

            bounding_box_breakpoints[layer] = []
            bounding_box_traces[layer] = []

            for i in range(len(localisation_data[layer])):

                bounding_box_breakpoints[layer].append([])
                bounding_box_traces[layer].append([0] * len(data))

        meta["layer_image_shape"] = layer_image_shape
        meta["bounding_box_breakpoints"] = bounding_box_breakpoints
        meta["bounding_box_traces"] = bounding_box_traces

        self.box_layer.metadata = meta
        self.plot_compute_progress.setValue(0)

        self.plot_graphs()
        self.plot_fit_graph()


    def initalise_gapseq_compute_traces(self):

        image_layers = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "localisation_threshold"]]

        for layer in image_layers:

            worker = Worker(partial(self.gapseq_compute_traces, layer = layer))
            worker.signals.result.connect(self.process_gapseq_compute_traces)
            worker.signals.progress.connect(partial(self.gapseq_progressbar, progressbar="compute"))
            self.threadpool.start(worker)

    def process_gapseq_undrift(self, image):

        self.viewer.layers["localisation_image"].data = image
        self.undrift_progress.setValue(0)

    def initialise_gapseq_undrift(self):

        img = self.viewer.layers["localisation_image"].data[0].copy()
        self.viewer.layers["localisation_image"].data = np.zeros_like(img)

        worker = Worker(self.gapseq_undrift)
        worker.signals.result.connect(self.process_gapseq_undrift)
        worker.signals.progress.connect(partial(self.gapseq_progressbar, progressbar="undrift"))
        self.threadpool.start(worker)

    def gapseq_undrift(self, progress_callback):

        n_frame_average = int(self.localisation_frame_averaging.currentText())
        meta = self.box_layer.metadata
        box_centres = meta["bounding_box_centres"]
        crop_mode = self.localisation_channel.currentIndex()
        stack_mode = self.localisation_stack_mode.currentIndex()
        path = self.viewer.layers["localisation_image"].metadata["image_path"]
        loc_data = []

        tif = tifffile.TiffFile(path)

        n_frames = len(tif.pages)
        n_segments = n_frames//n_frame_average
        image_index = np.split(np.arange(n_frames),n_segments)

        loc0_data = load_undrift_segment(image_index[0], path=path, box_centres=box_centres, crop_mode = crop_mode, stack_mode = stack_mode)

        with Pool() as p:

            def callback(*args):
                iter.append(1)
                progress = (len(iter) / n_segments) * 100

                if progress_callback != None:
                    progress_callback.emit(progress)

                return

            iter = []
            results = [p.apply_async(load_undrift_segment, args=(i,), kwds={'path': path,
                                                                            'box_centres': box_centres,
                                                                            'loc0_data': loc0_data,
                                                                            'crop_mode': crop_mode,
                                                                            'stack_mode': stack_mode}, callback=callback) for i in image_index]

            loc_data = [r.get() for r in results]

            drift = process_loc_data(loc_data, n_frames)

            index_list = np.arange(n_frames)
            iter = []
            images = [p.apply_async(undrift_image, args=(i,), kwds={'path': path,
                                                                     'drift': drift}, callback=callback) for i in index_list]

            images = [r.get() for r in images]

            index_list, images = zip(*images)
            index_list, images = zip(*sorted(zip(index_list, images), key=lambda x: x[0]))
            images = np.stack(images, axis=0)

            p.close()
            p.join()

        return images

    def update_cpd_controls(self):

        mode = self.fit_cpd_mode.currentIndex()

        if mode == 0:

            self.fit_cpd_breakpoints.setVisible(False)
            self.fit_cpd_breakpoints_label.setVisible(False)
            self.fit_cpd_min_size.setVisible(False)
            self.fit_cpd_min_size_label.setVisible(False)
            self.fit_cpd_window_size.setVisible(False)
            self.fit_cpd_window_size_label.setVisible(False)
            self.fit_cpd_jump.setVisible(False)
            self.fit_cpd_jump_label.setVisible(False)
            self.fit_cpd_penalty.setVisible(False)
            self.fit_cpd_penalty_label.setVisible(False)

        if mode == 1:

            self.fit_cpd_model_label.setVisible(True)
            self.fit_cpd_model.setVisible(True)
            self.fit_cpd_breakpoints_label.setVisible(False)
            self.fit_cpd_breakpoints.setVisible(False)
            self.fit_cpd_min_size_label.setVisible(True)
            self.fit_cpd_min_size.setVisible(True)
            self.fit_cpd_window_size_label.setVisible(False)
            self.fit_cpd_window_size.setVisible(False)
            self.fit_cpd_jump_label.setVisible(True)
            self.fit_cpd_jump.setVisible(True)
            self.fit_cpd_penalty_label.setVisible(True)
            self.fit_cpd_penalty.setVisible(True)


        if mode == 2:

            self.fit_cpd_breakpoints.setVisible(True)
            self.fit_cpd_breakpoints_label.setVisible(True)
            self.fit_cpd_min_size.setVisible(True)
            self.fit_cpd_min_size_label.setVisible(True)
            self.fit_cpd_window_size.setVisible(False)
            self.fit_cpd_window_size_label.setVisible(False)
            self.fit_cpd_jump.setVisible(True)
            self.fit_cpd_jump_label.setVisible(True)
            self.fit_cpd_penalty.setVisible(True)
            self.fit_cpd_penalty_label.setVisible(True)

        if mode == 3:

            self.fit_cpd_breakpoints.setVisible(True)
            self.fit_cpd_breakpoints_label.setVisible(True)
            self.fit_cpd_min_size.setVisible(True)
            self.fit_cpd_min_size_label.setVisible(True)
            self.fit_cpd_window_size.setVisible(True)
            self.fit_cpd_window_size_label.setVisible(True)
            self.fit_cpd_jump.setVisible(True)
            self.fit_cpd_jump_label.setVisible(True)
            self.fit_cpd_penalty.setVisible(True)
            self.fit_cpd_penalty_label.setVisible(True)

        if mode == 4:

            self.fit_cpd_breakpoints.setVisible(True)
            self.fit_cpd_breakpoints_label.setVisible(True)
            self.fit_cpd_min_size.setVisible(True)
            self.fit_cpd_min_size_label.setVisible(True)
            self.fit_cpd_window_size.setVisible(False)
            self.fit_cpd_window_size_label.setVisible(False)
            self.fit_cpd_jump.setVisible(True)
            self.fit_cpd_jump_label.setVisible(True)
            self.fit_cpd_penalty.setVisible(True)
            self.fit_cpd_penalty_label.setVisible(True)


        if mode == 5:

            self.fit_cpd_breakpoints.setVisible(True)
            self.fit_cpd_breakpoints_label.setVisible(True)
            self.fit_cpd_min_size.setVisible(True)
            self.fit_cpd_min_size_label.setVisible(True)
            self.fit_cpd_window_size.setVisible(False)
            self.fit_cpd_window_size_label.setVisible(False)
            self.fit_cpd_jump.setVisible(True)
            self.fit_cpd_jump_label.setVisible(True)
            self.fit_cpd_penalty.setVisible(False)
            self.fit_cpd_penalty_label.setVisible(False)

    def fit_graph_zoom(self,event,base_scale = 1.5):

        ax = event.inaxes
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location

        lines = ax.get_lines()
        line_xdata = lines[0].get_xdata()
        line_ydata = lines[0].get_ydata()

        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        if event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale

        xlim_min = xdata - cur_xrange*scale_factor
        xlim_max = xdata + cur_xrange*scale_factor

        if xlim_min < 0:
            xlim_min = 0
        if xlim_max > len(line_xdata):
            xlim_max = len(line_xdata)

        ydata_crop = np.array(line_ydata)[int(xlim_min):int(xlim_max)]

        ylim_min = np.min(ydata_crop)
        ylim_max = np.max(ydata_crop)
        ylim_range = ylim_max - ylim_min

        ylim_min -= ylim_range*0.1
        ylim_max += ylim_range*0.1

        ax.set_xlim([xlim_min,xlim_max])
        ax.set_ylim([ylim_min, ylim_max])

        self.fit_graph_canvas.draw()

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
                    self.fit_localisation_number.setMaximum(len(bounding_boxes) - 1)

                    self.plot_graphs()

    def keybind_classify_nucleotide(self, viewer, key):

        if "bounding_boxes" in self.viewer.layers:

            if "localisation_data" in self.box_layer.metadata.keys():

                localisation_number = self.plot_localisation_number.value()

                self.box_layer.metadata["nucleotide_class"][localisation_number] = str(key)

                self.plot_graphs()

    def keybind_classify_class(self, viewer, key):

        if "bounding_boxes" in self.viewer.layers:

            if "localisation_data" in self.box_layer.metadata.keys():

                localisation_number = self.plot_localisation_number.value()

                self.box_layer.metadata["bounding_box_class"][localisation_number] = int(key)

                self.plot_graphs()

    def fit_hmm(self, data, n_components, model = None, reorder = True):

        x = np.reshape(data, [len(data), 1])

        if model == None:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(n_components=n_components, n_iter=1000).fit(x)

        detected_states = model.predict(x)
        mus = np.array(model.means_)

        state_means = mus.flatten()
        state_index = np.arange(len(mus))

        state_means, state_index = zip(*sorted(zip(state_means, state_index), key=lambda x: x[0]))

        if reorder == True:

            reordered_states = np.zeros_like(detected_states)

            for i in range(len(np.unique(detected_states))):

                old_state = np.unique(detected_states)[i]
                new_state = np.where(state_index == old_state)[0][0]

                reordered_states[detected_states == old_state] = new_state

            hmm_states = reordered_states

        else:

            hmm_states = detected_states

        breakpoints = []
        for i in range(len(hmm_states) - 1):
            if hmm_states[i] != hmm_states[i + 1]:
                breakpoints.append(i)

        return hmm_states.tolist(), breakpoints

    def generate_cpd_trace(self,box_data,breakpoints, mode = "hmm", n_components = 2):

        if mode == "hmm":
            from hmmlearn.hmm import GaussianHMM
            x = np.reshape(box_data,[len(box_data),1])
            model = GaussianHMM(n_components=n_components, n_iter=1000).fit(x)

        if breakpoints == []:
            breakpoint_trace = [np.mean(box_data)]*len(box_data)
        else:

            if 0 not in breakpoints:
                breakpoints.insert(0,0)
            if len(x) not in breakpoints:
                breakpoints.append(len(x))

            breakpoint_trace = []

            breakpoints.sort()

            for i in range(len(breakpoints)-1):

                start = breakpoints[i]
                end = breakpoints[i+1]
                data = box_data[start:end].copy()

                if mode == "mean":
                    mean = np.mean(data)
                    value = [mean]*(end-start)
                    breakpoint_trace.extend(value)

                if mode == "hmm":
                    states, _ = self.fit_hmm(data,n_components, model= model)
                    vals, counts = np.unique(states, return_counts=True)
                    state = vals[np.argmax(counts)]
                    value = [state] * (end - start)
                    breakpoint_trace.extend(value)

        return breakpoint_trace

    def threaded_export_traces(self, mode="excel"):

        if "bounding_boxes" in self.viewer.layers:

            if len(self.box_layer.data) > 1:

                worker = Worker(partial(self.export_traces,mode=mode))
                worker.signals.progress.connect(partial(self.gapseq_progressbar, progressbar="export"))
                self.threadpool.start(worker)

    def export_traces(self, progress_callback, mode = "excel"):

        if "bounding_boxes" in self.viewer.layers:

            if "localisation_data" in self.box_layer.metadata.keys():

                meta = self.box_layer.metadata

                if "image_paths" in meta.keys():
                    path = meta["image_paths"][meta["image_layers"].index("localisation_image")]
                else:
                    path = self.viewer.layers["localisation_image"].metadata["image_path"]

                file_name = os.path.basename(path)
                directory = path.replace(file_name,"")

                extension = file_name.split(".")[-1]
                file_name = file_name.replace("."+extension,"_gapseq_traces.txt")

                if self.export_traces_at_import.isChecked() is False:

                    desktop = os.path.expanduser("~/Desktop")
                    directory = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

                if os.path.isdir(directory):

                    path = os.path.abspath(os.path.join(directory,file_name))

                    localisation_filter = self.traces_class_filter.currentText()
                    nucleotide_filter = self.traces_nucleotide_filter.currentText()
                    data_metric = self.traces_data_metric.currentIndex()
                    background_mode = self.traces_background_mode.currentIndex()

                    box_num = len(self.box_layer.data.copy())
                    meta = self.box_layer.metadata.copy()

                    bounding_box_class = meta["bounding_box_class"]
                    nucleotide_classes = meta["nucleotide_class"]
                    localisation_data = meta["localisation_data"]
                    bounding_box_breakpoints = meta["bounding_box_breakpoints"]
                    bounding_box_traces = meta["bounding_box_traces"]

                    layers = list(localisation_data.keys())

                    if self.traces_data_selection.currentIndex() == 0:
                        layers = [layer for layer in layers if layer == "localisation_image"]
                    if self.traces_data_selection.currentIndex() == 1:
                        layers = [layer for layer in layers if layer != "localisation_image"]

                    image_trace_index = []
                    image_trace_layer = []
                    image_trace_class = []
                    image_trace_nucleotide = []
                    image_trace_data = []

                    num_iter = len(layers) * int(box_num)
                    iter_count = 0

                    for i in range(box_num):

                        for j in range(len(layers)):

                            iter_count += 1
                            progress = int(iter_count / num_iter * 100)
                            progress_callback.emit(progress)

                            layer = layers[j]

                            box_data, _ = self.get_gapseq_trace_data(layer,i,data_metric,background_mode)

                            box_class = bounding_box_class[i]
                            nucleotide_class = nucleotide_classes[i]
                            breakpoints = bounding_box_breakpoints[layer][i]
                            trace = bounding_box_traces[layer][i]

                            if trace == []:

                                trace = [0]*len(box_data)

                            if localisation_filter == "None" and nucleotide_filter == "None":

                                image_trace_data.append(box_data)
                                image_trace_index.append(i)
                                image_trace_class.append(box_class)
                                image_trace_nucleotide.append(nucleotide_class)
                                image_trace_layer.append(layer)

                                if self.export_cpd_data.isChecked():

                                    image_trace_data.append(trace)
                                    image_trace_index.append(i)
                                    image_trace_class.append(box_class)
                                    image_trace_nucleotide.append(nucleotide_class)
                                    image_trace_layer.append(layer)

                            else:

                                if localisation_filter == "None" and nucleotide_filter == nucleotide_class:
                                    append_data = True
                                elif str(localisation_filter) == str(box_class) and nucleotide_filter == "None":
                                    append_data = True
                                elif str(localisation_filter) == str(box_class) and nucleotide_filter == nucleotide_class:
                                    append_data = True
                                else:
                                    append_data = False

                                if append_data is True:

                                    image_trace_data.append(box_data)
                                    image_trace_index.append(i)
                                    image_trace_class.append(box_class)
                                    image_trace_nucleotide.append(nucleotide_class)
                                    image_trace_layer.append(layer)

                                    if self.export_cpd_data.isChecked():

                                        breakpoint_trace = self.generate_cpd_trace(box_data, breakpoints)

                                        image_trace_data.append(breakpoint_trace)
                                        image_trace_index.append(i)
                                        image_trace_class.append(box_class)
                                        image_trace_nucleotide.append(nucleotide_class)
                                        image_trace_layer.append(layer)

                    if len(image_trace_data) > 0:

                        image_trace_data = np.stack(image_trace_data, axis=0).T

                        if mode == "dat":

                            path = path.replace(".txt",".dat")

                            image_trace_data = pd.DataFrame(image_trace_data,columns=image_trace_index)

                            image_trace_data.to_csv(path, sep=" ", index = False)

                        if mode == "excel":

                            path = path.replace(".txt", ".xlsx")

                            image_trace_data = pd.DataFrame(image_trace_data)
                            image_trace_data.columns = [image_trace_index,image_trace_layer,image_trace_class,image_trace_nucleotide]

                            with pd.ExcelWriter(path) as writer:
                                image_trace_data.to_excel(writer, sheet_name='Trace Data', index=True, startrow=1,startcol=1)

    def import_gapseq_data(self, mode = "all", path = None):

        if path == None:

            desktop = os.path.expanduser("~/Desktop")
            path, _ = QFileDialog.getOpenFileName(self, "Open Files", desktop, "GapSeq Files (*.txt)")

        if os.path.isfile(path):

            with open(path, 'r', encoding='utf-8') as f:
                gapseq_data = json.load(f, parse_int=int)

            bounding_boxes = gapseq_data["bounding_boxes"]
            image_layers = gapseq_data["image_layers"]
            image_paths = gapseq_data["image_paths"]
            image_metadata = gapseq_data["image_metadata"]
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
            meta["image_paths"] = gapseq_data["image_paths"]
            meta["image_metadata"] = gapseq_data["image_metadata"]
            meta["layer_image_shape"] = gapseq_data["layer_image_shape"]
            meta["nucleotide_class"] = gapseq_data["nucleotide_class"]
            meta["localisation_type"] = gapseq_data["localisation_type"]
            meta["bounding_box_breakpoints"] = gapseq_data["bounding_box_breakpoints"]
            meta["bounding_box_traces"] = gapseq_data["bounding_box_traces"]

            localisation_data = gapseq_data["localisation_data"]

            new_localisation_data = {}
            for layer_key,layer_value in localisation_data.items():
                new_localisation_data[layer_key] = {}
                for localisation_key, localisation_value in layer_value.items():
                    new_localisation_data[layer_key][int(localisation_key)] = localisation_value
            meta["localisation_data"] = new_localisation_data

            if "bounding_boxes" in self.viewer.layers:

                if meta["localisation_type"] == "Box":
                    self.box_layer.shape_type = ["Rectangle"] * len(bounding_boxes)

                if meta["localisation_type"] == "Circle":
                    self.box_layer.shape_type = ["Ellipse"] * len(bounding_boxes)

                self.viewer.layers["bounding_boxes"].data = bounding_boxes
                self.viewer.layers["bounding_boxes"].metadata = meta

            else:

                if meta["localisation_type"] == "Box":

                    self.box_layer = self.viewer.add_shapes(bounding_boxes, name="bounding_boxes", shape_type='Rectangle', edge_width=1, edge_color='red', face_color=[0, 0, 0, 0], opacity=0.3, metadata=meta)
                    self.box_layer.mouse_drag_callbacks.append(self.localisation_click_events)

                if meta["localisation_type"] == "Circle":

                    self.box_layer = self.viewer.add_shapes(bounding_boxes, name="bounding_boxes", shape_type='Ellipse',edge_width=1, edge_color='red', face_color=[0, 0, 0, 0],opacity=0.3,metadata=meta)
                    self.box_layer.mouse_drag_callbacks.append(self.localisation_click_events)

            self.sort_layer_order()

            self.plot_graphs()
            self.fit_plot_channel.addItems(image_layers)
            self.plot_fit_graph()

    def export_data(self):

        if "bounding_boxes" in self.viewer.layers:

            if "localisation_data" in self.box_layer.metadata.keys():

                bounding_boxes = self.box_layer.data.copy()

                meta = self.box_layer.metadata.copy()
#
                bounding_box_centres = meta["bounding_box_centres"]
                bounding_box_class = meta["bounding_box_class"]
                bounding_box_size = meta["bounding_box_size"]
                layer_image_shape = meta["layer_image_shape"]
                nucleotide_class = meta["nucleotide_class"]
                localisation_type = meta["localisation_type"]
                bounding_box_breakpoints = meta["bounding_box_breakpoints"]
                bounding_box_traces = meta["bounding_box_traces"]
                localisation_data = meta["localisation_data"]

                bounding_boxes = [box.tolist() for box in bounding_boxes]

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
                                       localisation_data=localisation_data,
                                       localisation_threshold = self.localisation_threshold.value(),
                                       bounding_box_size=bounding_box_size,
                                       image_layers = image_layers,
                                       image_paths = image_paths,
                                       image_metadata = image_metadata,
                                       layer_image_shape = layer_image_shape,
                                       nucleotide_class = nucleotide_class,
                                       localisation_type = localisation_type,
                                       bounding_box_breakpoints = bounding_box_breakpoints,
                                       bounding_box_traces = bounding_box_traces)

                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(gapseq_data, f, ensure_ascii=False, indent=4, cls = json_np_encoder)

    def get_background_mask(self, bounding_boxes,bounding_box_size, bounding_box_centres, image):

        bounding_boxes = self.box_layer.data.copy()
        shape_type = np.unique(self.box_layer.shape_type)[0]

        threshold_image = self.localisation_threshold_layer.data
        background_mask = threshold_image[:, threshold_image.shape[1] // 2:]

        kernel = np.ones((3, 3), np.uint8)
        background_mask = cv.dilate(background_mask, kernel, iterations=1)

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

    def compute_plot_data_mp(self):

        if "bounding_boxes" in self.viewer.layers:

            if len(self.box_layer.data) > 1:

                worker = Worker(self.compute_plot_data)
                worker.signals.result.connect(self.process_plot_data)
                worker.signals.progress.connect(partial(self.gapseq_progressbar, progressbar="compute"))
                self.threadpool.start(worker)

    def process_plot_data(self, plot_data):

        meta = self.box_layer.metadata.copy()

        meta["bounding_box_data"] = plot_data["bounding_box_data"]
        meta["layer_image_shape"] = plot_data["layer_image_shape"]
        meta["image_layers"] = plot_data["image_layers"]
        meta["background_data"] = plot_data["background_data"]
        meta["bounding_box_breakpoints"] = plot_data["bounding_box_breakpoints"]
        meta["bounding_box_traces"] = plot_data["bounding_box_traces"]

        self.box_layer.metadata = meta

        self.plot_graphs()
        self.plot_fit_graph()

    def compute_plot_data(self, progress_callback):

        image_layers = [layer.name for layer in self.viewer.layers if
                        layer.name not in ["bounding_boxes", "localisation_threshold"]]
        bounding_boxes = self.box_layer.data.copy()
        meta = self.box_layer.metadata.copy()

        bounding_box_centres = meta["bounding_box_centres"]
        bounding_box_class = meta["bounding_box_class"]
        bounding_box_size = meta["bounding_box_size"]
        layer_image_shape = {}
        bounding_box_data = {}
        bounding_box_breakpoints = {}
        bounding_box_traces = {}
        background_data = {}

        num_iter = len(image_layers)*len(bounding_boxes)
        iter_count = 0

        for i in range(len(image_layers)):

            try:

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    image = self.viewer.layers[image_layers[i]].data
                    layer = image_layers[i]
                    bounding_box_data[layer] = []
                    bounding_box_breakpoints[layer] = []
                    bounding_box_traces[layer] = []
                    layer_image_shape[layer] = image.shape

                    background_image, masked_image = self.get_background_mask(bounding_boxes, bounding_box_size, bounding_box_centres, image)

                    background_data[layer]= {"global_background": np.mean(background_image,axis=(1, 2)).tolist(),
                                             "local_background": []}

                    for j in range(len(bounding_boxes)):

                        iter_count += 1

                        progress = int(iter_count/num_iter * 100)
                        progress_callback.emit(progress)

                        polygon = bounding_boxes[j]
                        [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = polygon
                        cx,cy = bounding_box_centres[j]
                        box_class = bounding_box_class[j]
                        box_size = bounding_box_size
                        background_box_size = 20

                        data = image[:, int(y1):int(y2), int(x1):int(x2)]
                        data = np.nanmean(data, axis=(1, 2)).tolist()

                        [[y1,x1],[y2,x2]] = [[cy - background_box_size, cx - background_box_size],
                                             [cy + background_box_size, cx + background_box_size]]
                        local_background_data = background_image[:, int(y1):int(y2), int(x1):int(x2)].copy()
                        local_background_data = np.mean(local_background_data, axis=(1, 2)).tolist()

                        bounding_box_data[layer].append(data)
                        bounding_box_breakpoints[layer].append([])
                        bounding_box_traces[layer].append([0]*len(data))
                        background_data[layer]["local_background"].append(local_background_data)

            except:
                pass

        compute_data = dict(bounding_box_data=bounding_box_data,
                            layer_image_shape=layer_image_shape,
                            image_layers=image_layers,
                            background_data=background_data,
                            bounding_box_breakpoints=bounding_box_breakpoints,
                            bounding_box_traces=bounding_box_traces)


        return compute_data

    def gapseq_progressbar(self, progress, progressbar):

        if progressbar == "compute":
            self.plot_compute_progress.setValue(progress)
        if progressbar == "fit_traces":
            self.fit_traces_progress.setValue(progress)
        if progressbar == "export":
            self.export_progress.setValue(progress)
        if progressbar == "undrift":
            self.undrift_progress.setValue(progress)

        if progress == 100:
            time.sleep(1)
            self.plot_compute_progress.setValue(100)
            self.fit_traces_progress.setValue(100)
            self.export_progress.setValue(100)
            self.undrift_progress.setValue(100)
            time.sleep(1)
            self.plot_compute_progress.setValue(0)
            self.fit_traces_progress.setValue(0)
            self.export_progress.setValue(0)
            self.undrift_progress.setValue(100)

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

    def manual_state_edit(self, event):

        if event.xdata != None:

            key = event.key

            if key.isdigit():

                localisation_number = self.fit_localisation_number.value()

                frame_int = int(event.xdata)
                ax = self.fit_graph_canvas.figure.axes[0]

                lines = ax.get_lines()
                layer = [str(line.get_label()) for line in lines if str(line.get_label()) != "Frame"][0]

                meta = self.box_layer.metadata.copy()

                break_points = meta["bounding_box_breakpoints"][layer][localisation_number]
                bounding_box_trace = meta["bounding_box_traces"][layer][localisation_number]

                start = break_points[np.max(np.where(np.array(break_points) < frame_int))]
                end = break_points[np.min(np.where(np.array(break_points) > frame_int))]

                bounding_box_trace[start:end] = [int(key)] * (end-start)

                meta["bounding_box_traces"][layer][localisation_number] = bounding_box_trace.tolist()

                self.box_layer.metadata = meta
                self.plot_fit_graph()

    def manual_break_point_edit(self, event):

        if event.xdata != None:

            localisation_number = self.fit_localisation_number.value()

            frame_int = int(event.xdata)
            ax = self.fit_graph_canvas.figure.axes[0]

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            lines = ax.get_lines()
            layer = [str(line.get_label()) for line in lines if str(line.get_label()) != "Frame"][0]

            meta = self.box_layer.metadata.copy()
            data, _ = self.get_gapseq_trace_data(layer,localisation_number)

            hmm_states = self.fit_hmm_states.value()

            if "bounding_box_breakpoints" in meta.keys():
                break_points = meta["bounding_box_breakpoints"][layer][localisation_number]
            else:
                break_points = []

            if event.button == 1 and event.xdata != None:

                break_points.append(frame_int)

                bounding_box_trace = self.generate_cpd_trace(data,break_points,mode="hmm",n_components=hmm_states)

                meta["bounding_box_breakpoints"][layer][localisation_number] = break_points
                meta["bounding_box_traces"][layer][localisation_number] = bounding_box_trace

                self.box_layer.metadata = meta
                self.plot_fit_graph(xlim=xlim,ylim=ylim)

            if event.button == 3 and event.xdata != None:

                if len(break_points) > 0:

                    index = (np.abs(np.array(break_points) - frame_int)).argmin()
                    closest_value = break_points[index]

                    distance = abs(closest_value - frame_int)

                    if distance < 20:

                        del break_points[index]

                        bounding_box_trace = self.generate_cpd_trace(data,break_points,mode="hmm",n_components=hmm_states)
                        meta["bounding_box_breakpoints"][layer][localisation_number] = break_points
                        meta["bounding_box_traces"][layer][localisation_number] = bounding_box_trace

                        self.box_layer.metadata = meta
                        self.plot_fit_graph(xlim=xlim,ylim=ylim)

    def change_point_detection_mp(self,detection_mode):

        if "bounding_boxes" in self.viewer.layers:

            worker = Worker(partial(self.change_point_detection, detection_mode=detection_mode))
            worker.signals.progress.connect(partial(self.gapseq_progressbar, progressbar="fit_traces"))
            self.threadpool.start(worker)

    def change_point_detection(self, progress_callback, detection_mode):

        if "bounding_boxes" in self.viewer.layers:

            if "localisation_data" in self.box_layer.metadata.keys():

                bounding_boxes = self.box_layer.data.copy()
                meta = self.box_layer.metadata.copy()
                mode = self.fit_cpd_mode.currentIndex()
                hmm_states = self.fit_hmm_states.value()

                if detection_mode == "active":
                    localisation_list = [self.fit_localisation_number.value()]
                    layer_list = [self.fit_plot_channel.currentText()]
                else:
                    localisation_list = np.arange((len(bounding_boxes)))
                    layer_list = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "localisation_threshold"]]

                num_iter = len(layer_list) * len(localisation_list)
                iter_count = 0

                for layer in layer_list:

                    for i in range(len(localisation_list)):

                        localisation_number = localisation_list[i]

                        iter_count += 1
                        progress = int((iter_count/num_iter)*100)
                        progress_callback.emit(progress)

                        try:
                            plot_metric_index = self.fit_plot_metric.currentIndex()
                            background_subtraction_mode = self.plot_background_subtraction_mode.currentIndex()

                            points, _ = self.get_gapseq_trace_data(layer,
                                                                    localisation_number,
                                                                    plot_metric_index,
                                                                    background_subtraction_mode)
                            if points != None:

                                model = self.fit_cpd_model.currentText()
                                pen = self.fit_cpd_penalty.value()
                                width = self.fit_cpd_window_size.value()
                                min_size = self.fit_cpd_min_size.value()
                                jump = self.fit_cpd_jump.value()
                                n_bkps = self.fit_cpd_breakpoints.value()

                                if mode == 0:
                                    bounding_box_trace, breakpoints = self.fit_hmm(points,hmm_states)
                                if mode == 1:
                                    algo = rpt.Pelt(model=model, jump=jump).fit(points)
                                    breakpoints = algo.predict(pen=pen)
                                if mode == 2:
                                    algo = rpt.Binseg(model=model,min_size=min_size, jump=jump).fit(points)
                                    breakpoints =algo.predict(n_bkps=n_bkps, pen=pen)
                                if mode == 3:
                                    algo = rpt.Window(width=width, model=model, jump=jump).fit(points)
                                    breakpoints = algo.predict(n_bkps=n_bkps, pen=pen)
                                if mode == 4:
                                    algo = rpt.BottomUp(model=model, jump=jump, min_size=min_size).fit(points)
                                    breakpoints = algo.predict(n_bkps=n_bkps, pen=pen)
                                if mode == 5:
                                    algo = rpt.Dynp(model=model, min_size=min_size, jump=jump).fit(points)
                                    breakpoints = algo.predict(n_bkps=n_bkps)

                                if mode != 0:
                                    bounding_box_trace = self.generate_cpd_trace(points, breakpoints, mode = "hmm", n_components = hmm_states)

                                meta["bounding_box_breakpoints"][layer][localisation_number] = breakpoints
                                meta["bounding_box_traces"][layer][localisation_number] = bounding_box_trace

                        except:
                            print(traceback.format_exc())
                            pass

                self.box_layer.metadata = meta
                self.plot_fit_graph()

    def plot_fit_graph(self, plot_data = None, xlim = None, ylim = None):

        if "bounding_boxes" in self.viewer.layers:

            if "localisation_data" in self.box_layer.metadata.keys():

                self.fit_localisation_number.setMaximum(len(self.box_layer.data) - 1)

                plot_data = self.get_fit_graph_data()

                if plot_data != None:

                    x = plot_data["x"]
                    y = plot_data["y"]

                    if self.fit_plot_normalise.isChecked():
                        y = (y - np.min(y)) / (np.max(y) - np.min(y))

                    localisation_number = plot_data["localisation_number"]
                    bounding_box_trace = plot_data["bounding_box_trace"]

                    self.fit_graph_canvas.figure.clf()
                    axes = self.fit_graph_canvas.figure.add_subplot(111)
                    axes.set_facecolor("#262930")
                    axes.plot(x, y, label=plot_data["layer"])

                    if xlim != None:
                        axes.set_xlim(xlim)
                    else:
                        axes.set_xlim(plot_data["x_min"], plot_data["x_max"])

                    if ylim != None:
                        axes.set_ylim(ylim)

                    break_points = plot_data["break_points"]

                    if len(break_points) > 0 and self.show_cpd_breakpoints.isChecked():
                        axes.vlines(break_points, ymin = plot_data["y_min"], ymax=plot_data["y_max"], colors="red",label="Change Points")

                    if self.show_cpd_states.isChecked():
                        axes2 = axes.twinx()
                        axes2.plot(bounding_box_trace, color='blue',label="States")

                    self.fit_graph_canvas.figure.tight_layout()
                    self.fit_graph_canvas.draw()

    def get_fit_graph_data(self, layer = None, localisation_number = None, background_subtraction_mode = None):

        if "bounding_boxes" in self.viewer.layers:

            if "localisation_data" in self.box_layer.metadata.keys():

                try:

                    if localisation_number is None:
                        localisation_number = self.fit_localisation_number.value()

                    if layer is None:
                        layer = self.fit_plot_channel.currentText()

                    if background_subtraction_mode is None:
                        background_subtraction_mode = self.fit_background_subtraction_mode.currentIndex()

                    plot_metric_index = self.fit_plot_metric.currentIndex()

                    data, bounding_box_data = self.get_gapseq_trace_data(layer,
                                                                         localisation_number,
                                                                         plot_metric_index,
                                                                         background_subtraction_mode)

                    if "bounding_box_breakpoints" in self.box_layer.metadata.keys():
                        meta = self.box_layer.metadata
                        break_points = meta["bounding_box_breakpoints"][layer][localisation_number]
                        bounding_box_trace = meta["bounding_box_traces"][layer][localisation_number]
                    else:
                        break_points = []
                        bounding_box_trace = [0]*len(data)

                    x_min = 0
                    x_max = len(data)

                    x = np.arange(len(data))
                    y = data

                    plot_data = dict(x=x,y=y,localisation_number=localisation_number,
                                     x_min=x_min,x_max=x_max,
                                     y_min = np.min(y), y_max = np.max(y),
                                     layer = layer, break_points = break_points,
                                     bounding_box_trace = bounding_box_trace)

                except:
                    plot_data = None
                    print(traceback.format_exc())

        else:
            plot_data = None

        return plot_data

    def plot_graphs(self):

        plot_data = None
        plot_mode_index = self.plot_mode.currentIndex()

        meta = self.box_layer.metadata

        if "bounding_boxes" in self.viewer.layers:

            if "localisation_data" in meta.keys():

                image_layers = meta["localisation_data"].keys()

                if plot_mode_index == 0:

                    layers = ["localisation_image"]

                    self.akseg_maxmimum_height = int(self.akseg_ui.frameGeometry().height()*0.6)
                    maximum_height = 400 * len(layers)
                    if maximum_height > self.akseg_maxmimum_height:
                        maximum_height = self.akseg_maxmimum_height
                    self.graph_container.setMinimumHeight(maximum_height)

                    frame_num = np.max([meta["localisation_data"][layer][0]["image_shape"][0] for layer in layers])

                    self.plot_frame_number.setMaximum(frame_num-1)
                    current_step = list(self.viewer.dims.current_step)
                    current_step[0] = self.plot_frame_number.value()
                    self.viewer.dims.current_step = tuple(current_step)

                    plot_data = self.get_plot_data(layers=["localisation_image"])
                    self.plot(plot_data=plot_data)

                if plot_mode_index == 1:

                    layers = [layer for layer in image_layers if layer not in ["localisation_image", "localisation_threshold", "bounding_boxes"]]

                    self.akseg_maxmimum_height = int(self.akseg_ui.frameGeometry().height()*0.6)
                    maximum_height = 400 * len(layers)
                    if maximum_height > self.akseg_maxmimum_height:
                        maximum_height = self.akseg_maxmimum_height
                    self.graph_container.setMinimumHeight(maximum_height)

                    if len(layers) > 0:

                        frame_num = np.max([meta["localisation_data"][layer][0]["image_shape"][0] for layer in layers])

                        self.plot_frame_number.setMaximum(frame_num - 1)
                        current_step = list(self.viewer.dims.current_step)
                        current_step[0] = self.plot_frame_number.value()
                        self.viewer.dims.current_step = tuple(current_step)

                        layers = sorted(layers)

                        plot_data = self.get_plot_data(layers=layers)
                        self.plot(plot_data=plot_data)

                if plot_mode_index == 2:

                    layers = [layer for layer in image_layers if layer not in ["bounding_boxes", "localisation_threshold"]]

                    self.akseg_maxmimum_height = int(self.akseg_ui.frameGeometry().height()*0.6)
                    maximum_height = 400 * len(layers)
                    if maximum_height > self.akseg_maxmimum_height:
                        maximum_height = self.akseg_maxmimum_height
                    self.graph_container.setMinimumHeight(maximum_height)

                    if len(layers) > 0:

                        layers = sorted(layers)
                        layers.insert(0,layers.pop(layers.index("localisation_image")))

                        frame_num = np.max([meta["localisation_data"][layer][0]["image_shape"][0] for layer in layers])

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

    def get_gapseq_trace_data(self, layer, localisation_number, plot_metric_index = 0, background_subtraction_mode = 0):

        plot_metric_dict = {0: "box_mean", 1: "box_std", 2: "box_range",
                            3: "gaussian_height", 4: "gausian_width"}

        plot_metric = plot_metric_dict[plot_metric_index]

        localisation_data = self.box_layer.metadata["localisation_data"]

        bounding_box_data = localisation_data[layer][localisation_number]

        data = bounding_box_data[plot_metric]

        if background_subtraction_mode == 1 and plot_metric + '_local_background' in bounding_box_data.keys():
            background = bounding_box_data[plot_metric + '_local_background']
            data = list(np.array(data) - np.array(background))
            data = list(data - np.min(data))

        if background_subtraction_mode == 2 and plot_metric + '_global_background' in bounding_box_data.keys():
            background = bounding_box_data[plot_metric + '_global_background']
            data = np.array(data) - np.array(background)
            data = list(data - np.min(data))

        return data, bounding_box_data

    def get_plot_data(self, layers = ["localisation_image"]):

        localisation_filter = self.plot_localisation_filter.currentText()
        nucleotide_filter = self.plot_nucleotide_filter.currentText()

        if "localisation_data" in self.box_layer.metadata.keys():

            meta = self.box_layer.metadata.copy()

            localisation_data = meta["localisation_data"]

            plot_data = []

            for layer in layers:

                if localisation_filter == "None" and nucleotide_filter == "None":

                    self.plot_localisation_number.setMaximum(len(self.box_layer.data) - 1)

                    frame_number = self.plot_frame_number.value()
                    localisation_number = self.plot_localisation_number.value()

                    if localisation_number != -1:

                        bounding_box_class = self.box_layer.metadata["bounding_box_class"][localisation_number]
                        nucleotide_class = self.box_layer.metadata["nucleotide_class"][localisation_number]

                        plot_metric_index = self.plot_metric.currentIndex()
                        background_subtraction_mode = self.plot_background_subtraction_mode.currentIndex()

                        data, bounding_box_data = self.get_gapseq_trace_data(layer,
                                                                             localisation_number,
                                                                             plot_metric_index,
                                                                             background_subtraction_mode)

                        image_shape = bounding_box_data["image_shape"]

                        [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = bounding_box_data["bounding_box"]

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

                            if localisation_number >= 0:

                                bounding_box_class = self.box_layer.metadata["bounding_box_class"][localisation_number]
                                nucleotide_class = self.box_layer.metadata["nucleotide_class"][localisation_number]

                                plot_metric_index = self.plot_metric.currentIndex()
                                background_subtraction_mode = self.plot_background_subtraction_mode.currentIndex()

                                data, bounding_box_data = self.get_gapseq_trace_data(layer,
                                                                                     localisation_number,
                                                                                     plot_metric_index,
                                                                                     background_subtraction_mode)

                                image_shape = bounding_box_data["image_shape"]

                                [[y2, x1], [y1, x1], [y2, x2], [y1, x2]] = bounding_box_data["bounding_box"]

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

                if self.plot_normalise.isChecked():
                    y = (y - np.min(y)) / (np.max(y) - np.min(y))

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

        if "Control" in event.modifiers:

            selected_layer = self.viewer.layers.selection.active
            data_coordinates = selected_layer.world_to_data(event.position)
            coord = np.round(data_coordinates).astype(int)
            self.current_coord = coord

            shape_type = np.unique(self.box_layer.shape_type)

            if len(shape_type) == 0:
                shape_type = self.localisation_type.currentText()
            else:
                shape_type = shape_type[0]

            if coord is not None:

                if len(coord) > 2:
                    coord = coord[1:]

                bounding_boxes = self.box_layer.data
                meta = self.box_layer.metadata.copy()

                bounding_box_centres = meta["bounding_box_centres"]
                bounding_box_class = meta["bounding_box_class"]
                bounding_box_size = meta["bounding_box_size"]
                nucleotide_class = meta["nucleotide_class"]

                localisation_type = self.localisation_type.currentText()


                box_index = self.box_layer.get_value(coord)[0]

                if box_index is not None:

                    if len(bounding_boxes) > 0:

                        del bounding_boxes[box_index]
                        del bounding_box_centres[box_index]
                        del bounding_box_class[box_index]
                        del nucleotide_class[box_index]

                        meta["bounding_box_centres"] = bounding_box_centres
                        meta["bounding_box_class"] = bounding_box_class
                        meta["nucleotide_class"] = nucleotide_class

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
                    bounding_box_class.append(0)
                    nucleotide_class.append("N/A")

                    meta["bounding_box_size"] = size
                    meta["bounding_box_centres"] = bounding_box_centres
                    meta["bounding_box_class"] = bounding_box_class
                    meta["nucleotide_class"] = nucleotide_class
                    meta["localisation_type"] = self.localisation_type.currentText()

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
            background_bounding_boxes = []
            bounding_circles = []
            bounding_box_centres = []
            bounding_box_class = []
            nucleotide_class = []

            for i in range(len(contours)):

                try:

                    cnt = contours[i]

                    x, y, w, h = cv.boundingRect(cnt)

                    area = cv2.contourArea(cnt)

                    M = cv.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    size = bounding_box_size

                    polygon = [[cy - size, cx - size],
                               [cy + size, cx - size],
                               [cy + size, cx + size],
                               [cy - size, cx + size]]

                    if area > localisation_area_min_value and area < localisation_area_max_value:

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

                self.box_layer.data = bounding_boxes
                self.box_layer.metadata = meta

                if self.localisation_type.currentText() == "Box":
                    self.box_layer.shape_type = ["Rectangle"]*len(bounding_boxes)

                if self.localisation_type.currentText() == "Circle":
                    self.box_layer.shape_type = ["Ellipse"]*len(bounding_boxes)

                self.fit_localisations()
                self.box_layer.mouse_drag_callbacks.append(self.localisation_click_events)

                # pd.DataFrame(bounding_box_centres).to_csv("dev/bounding_boxes.txt")

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

    def fitgaussian(self, data, params = []):

        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""

        if len(params) == 0:
            params = self.moments(data)

        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)

        return p

    def stack_image(self, image):

        stack_mode = self.localisation_stack_mode.currentIndex()

        if stack_mode == 0:

            image = np.mean(image,axis=0)

        if stack_mode == 1:

            image = np.max(image,axis=0)

        if stack_mode == 2:

            image = np.std(image,axis=0)

        return image


    def import_localisation_image(self, meta = [], import_gapseq = False):

        # path = r"C:/napari-gapseq/src/napari_gapseq/dev/20220527_27thMay2022GAP36A/27thMay2022GAPSeq4onebyonesubstrategAPGS8FRETfoursea50nMconce2_GAPSeqonebyoneGAP36AL532Exp200.tif"
        # # path = r"C:/napari-gapseq/src/napari_gapseq/dev/20220527_27thMay2022GAP36A/27thMay2022GAPSeq4onebyonesubstrategAPGS8FRETfoursea50nMconce2_GAPSeqonebyoneGAP36AL532Exp200_locImage.tiff"
        # crop_mode = self.localisation_channel.currentIndex()
        # localisation_threshold = self.localisation_threshold.value()

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

                img = self.stack_image(image).astype(np.uint8)

                img = difference_of_gaussians(img, 1)

                _, localisation_mask = cv.threshold(img, localisation_threshold, 255, cv.THRESH_BINARY)

                footprint = disk(1)
                localisation_mask = erosion(localisation_mask, footprint)

                localisation_threshold_image = np.hstack((img,localisation_mask))

                meta["localisation_threshold"] = localisation_threshold
                meta["path"] = path

            else:

                image, meta = self.read_image_file(path,crop_mode)
                meta["crop_mode"] = crop_mode

                img = self.stack_image(image)

                img = difference_of_gaussians(img, 1)

                img = normalize99(img)
                img = rescale01(img) * 255
                img = img.astype(np.uint8)

                # localisation_mask_image = cv.fastNlMeansDenoising(img, h=30, templateWindowSize=5, searchWindowSize=31)

                _, localisation_mask = cv.threshold(img, localisation_threshold, 255, cv.THRESH_BINARY)

                footprint = disk(1)
                localisation_mask = erosion(localisation_mask, footprint)

                localisation_threshold_image = np.hstack((img,localisation_mask))

                meta["localisation_threshold"] = localisation_threshold
                meta["path"] = path

            meta["bounding_box_centres"] = []
            meta["bounding_box_class"] = []
            meta["bounding_box_size"] = []
            meta["nucleotide_class"] = []
            meta["localisation_type"] = []
            meta["background_data"] = {"global_background": [],"local_background": []}

            if "localisation_image" in self.viewer.layers:

                self.localisation_image_layer.data = image
                self.localisation_image_layer.metadata = meta
                self.localisation_threshold_layer.data = localisation_threshold_image
                self.localisation_threshold_layer.metadata = meta

            else:

                if "bounding_boxes" not in self.viewer.layers:

                    if self.localisation_type.currentText() == "Box":
                        self.box_layer = self.viewer.add_shapes(name="bounding_boxes", shape_type='Rectangle',edge_width=1, edge_color='red', face_color=[0, 0, 0, 0],opacity=0.3, metadata=meta)

                    if self.localisation_type.currentText() == "Circle":
                        self.box_layer = self.viewer.add_shapes(name="bounding_boxes", shape_type='Ellipse',edge_width=1, edge_color='red', face_color=[0, 0, 0, 0],opacity=0.3, metadata=meta)

                self.localisation_image_layer = self.viewer.add_image(image, name="localisation_image",metadata=meta)
                self.localisation_threshold_layer = self.viewer.add_image(localisation_threshold_image, name="localisation_threshold", metadata=meta)

                self.localisation_image_layer.mouse_drag_callbacks.append(self.localisation_click_events)
                self.localisation_threshold_layer.mouse_drag_callbacks.append(self.localisation_click_events)

            self.plot_frame_number.setMaximum(image.shape[0]-1)
            self.fit_plot_channel.addItems(["localisation_image"])
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
            self.fit_plot_channel.addItem(layer_name)

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
