# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gapseq_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TabWidget(object):
    def setupUi(self, TabWidget):
        TabWidget.setObjectName("TabWidget")
        TabWidget.resize(371, 727)
        self.tab_localisations = QtWidgets.QWidget()
        self.tab_localisations.setObjectName("tab_localisations")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_localisations)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.tab_localisations)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.localisation_channel = QtWidgets.QComboBox(self.tab_localisations)
        self.localisation_channel.setObjectName("localisation_channel")
        self.localisation_channel.addItem("")
        self.localisation_channel.addItem("")
        self.localisation_channel.addItem("")
        self.localisation_channel.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.localisation_channel)
        self.localisation_import_image = QtWidgets.QPushButton(self.tab_localisations)
        self.localisation_import_image.setObjectName("localisation_import_image")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.localisation_import_image)
        self.verticalLayout_2.addLayout(self.formLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem)
        self.label_11 = QtWidgets.QLabel(self.tab_localisations)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_2.addWidget(self.label_11)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_7 = QtWidgets.QLabel(self.tab_localisations)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 0, 0, 1, 1)
        self.localisation_threshold_label = QtWidgets.QLabel(self.tab_localisations)
        self.localisation_threshold_label.setMinimumSize(QtCore.QSize(20, 0))
        self.localisation_threshold_label.setBaseSize(QtCore.QSize(5, 0))
        self.localisation_threshold_label.setObjectName("localisation_threshold_label")
        self.gridLayout_2.addWidget(self.localisation_threshold_label, 0, 2, 1, 1)
        self.localisation_threshold = QtWidgets.QSlider(self.tab_localisations)
        self.localisation_threshold.setMaximum(255)
        self.localisation_threshold.setProperty("value", 200)
        self.localisation_threshold.setOrientation(QtCore.Qt.Horizontal)
        self.localisation_threshold.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.localisation_threshold.setObjectName("localisation_threshold")
        self.gridLayout_2.addWidget(self.localisation_threshold, 0, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem1)
        self.label_6 = QtWidgets.QLabel(self.tab_localisations)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_27 = QtWidgets.QLabel(self.tab_localisations)
        self.label_27.setObjectName("label_27")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_27)
        self.localisation_type = QtWidgets.QComboBox(self.tab_localisations)
        self.localisation_type.setObjectName("localisation_type")
        self.localisation_type.addItem("")
        self.localisation_type.addItem("")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.localisation_type)
        self.verticalLayout_2.addLayout(self.formLayout_5)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.localisation_aspect_ratio_label = QtWidgets.QLabel(self.tab_localisations)
        self.localisation_aspect_ratio_label.setMinimumSize(QtCore.QSize(20, 0))
        self.localisation_aspect_ratio_label.setObjectName("localisation_aspect_ratio_label")
        self.gridLayout.addWidget(self.localisation_aspect_ratio_label, 3, 2, 1, 1)
        self.localisation_area_min = QtWidgets.QSlider(self.tab_localisations)
        self.localisation_area_min.setMaximum(100)
        self.localisation_area_min.setProperty("value", 0)
        self.localisation_area_min.setOrientation(QtCore.Qt.Horizontal)
        self.localisation_area_min.setObjectName("localisation_area_min")
        self.gridLayout.addWidget(self.localisation_area_min, 1, 1, 1, 1)
        self.localisation_area_min_label = QtWidgets.QLabel(self.tab_localisations)
        self.localisation_area_min_label.setEnabled(True)
        self.localisation_area_min_label.setMinimumSize(QtCore.QSize(20, 0))
        self.localisation_area_min_label.setBaseSize(QtCore.QSize(5, 0))
        self.localisation_area_min_label.setObjectName("localisation_area_min_label")
        self.gridLayout.addWidget(self.localisation_area_min_label, 1, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.tab_localisations)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 2, 0, 1, 1)
        self.localisation_area_max = QtWidgets.QSlider(self.tab_localisations)
        self.localisation_area_max.setMaximum(100)
        self.localisation_area_max.setProperty("value", 7)
        self.localisation_area_max.setOrientation(QtCore.Qt.Horizontal)
        self.localisation_area_max.setObjectName("localisation_area_max")
        self.gridLayout.addWidget(self.localisation_area_max, 2, 1, 1, 1)
        self.localisation_area_max_label = QtWidgets.QLabel(self.tab_localisations)
        self.localisation_area_max_label.setEnabled(True)
        self.localisation_area_max_label.setMinimumSize(QtCore.QSize(20, 0))
        self.localisation_area_max_label.setBaseSize(QtCore.QSize(5, 0))
        self.localisation_area_max_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.localisation_area_max_label.setObjectName("localisation_area_max_label")
        self.gridLayout.addWidget(self.localisation_area_max_label, 2, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.tab_localisations)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 0, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.tab_localisations)
        self.label_28.setObjectName("label_28")
        self.gridLayout.addWidget(self.label_28, 3, 0, 1, 1)
        self.localisation_aspect_ratio = QtWidgets.QSlider(self.tab_localisations)
        self.localisation_aspect_ratio.setMinimum(10)
        self.localisation_aspect_ratio.setMaximum(20)
        self.localisation_aspect_ratio.setPageStep(1)
        self.localisation_aspect_ratio.setProperty("value", 15)
        self.localisation_aspect_ratio.setOrientation(QtCore.Qt.Horizontal)
        self.localisation_aspect_ratio.setObjectName("localisation_aspect_ratio")
        self.gridLayout.addWidget(self.localisation_aspect_ratio, 3, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.localisation_detect = QtWidgets.QPushButton(self.tab_localisations)
        self.localisation_detect.setObjectName("localisation_detect")
        self.verticalLayout_2.addWidget(self.localisation_detect)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem2)
        self.label_12 = QtWidgets.QLabel(self.tab_localisations)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_2.addWidget(self.label_12)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_10 = QtWidgets.QLabel(self.tab_localisations)
        self.label_10.setObjectName("label_10")
        self.gridLayout_3.addWidget(self.label_10, 0, 0, 1, 1)
        self.localisation_bbox_size = QtWidgets.QSlider(self.tab_localisations)
        self.localisation_bbox_size.setAutoFillBackground(False)
        self.localisation_bbox_size.setMaximum(10)
        self.localisation_bbox_size.setProperty("value", 4)
        self.localisation_bbox_size.setOrientation(QtCore.Qt.Horizontal)
        self.localisation_bbox_size.setObjectName("localisation_bbox_size")
        self.gridLayout_3.addWidget(self.localisation_bbox_size, 0, 1, 1, 1)
        self.localisation_bbox_size_label = QtWidgets.QLabel(self.tab_localisations)
        self.localisation_bbox_size_label.setEnabled(True)
        self.localisation_bbox_size_label.setMinimumSize(QtCore.QSize(20, 0))
        self.localisation_bbox_size_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.localisation_bbox_size_label.setObjectName("localisation_bbox_size_label")
        self.gridLayout_3.addWidget(self.localisation_bbox_size_label, 0, 2, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_3)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem3)
        TabWidget.addTab(self.tab_localisations, "")
        self.tab_images = QtWidgets.QWidget()
        self.tab_images.setObjectName("tab_images")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_images)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_13 = QtWidgets.QLabel(self.tab_images)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_3.addWidget(self.label_13)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_14 = QtWidgets.QLabel(self.tab_images)
        self.label_14.setObjectName("label_14")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.image_import_channel = QtWidgets.QComboBox(self.tab_images)
        self.image_import_channel.setObjectName("image_import_channel")
        self.image_import_channel.addItem("")
        self.image_import_channel.addItem("")
        self.image_import_channel.addItem("")
        self.image_import_channel.addItem("")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.image_import_channel)
        self.label_15 = QtWidgets.QLabel(self.tab_images)
        self.label_15.setObjectName("label_15")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.image_gap_code = QtWidgets.QComboBox(self.tab_images)
        self.image_gap_code.setObjectName("image_gap_code")
        self.image_gap_code.addItem("")
        self.image_gap_code.addItem("")
        self.image_gap_code.addItem("")
        self.image_gap_code.addItem("")
        self.image_gap_code.addItem("")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.image_gap_code)
        self.label_16 = QtWidgets.QLabel(self.tab_images)
        self.label_16.setObjectName("label_16")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.image_sequence_code = QtWidgets.QComboBox(self.tab_images)
        self.image_sequence_code.setObjectName("image_sequence_code")
        self.image_sequence_code.addItem("")
        self.image_sequence_code.addItem("")
        self.image_sequence_code.addItem("")
        self.image_sequence_code.addItem("")
        self.image_sequence_code.addItem("")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.image_sequence_code)
        self.verticalLayout_3.addLayout(self.formLayout_4)
        self.import_image = QtWidgets.QPushButton(self.tab_images)
        self.import_image.setObjectName("import_image")
        self.verticalLayout_3.addWidget(self.import_image)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem4)
        TabWidget.addTab(self.tab_images, "")
        self.tab_plots = QtWidgets.QWidget()
        self.tab_plots.setObjectName("tab_plots")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_plots)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_22 = QtWidgets.QLabel(self.tab_plots)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.verticalLayout_4.addWidget(self.label_22)
        self.plot_compute = QtWidgets.QPushButton(self.tab_plots)
        self.plot_compute.setObjectName("plot_compute")
        self.verticalLayout_4.addWidget(self.plot_compute)
        self.plot_compute_progress = QtWidgets.QProgressBar(self.tab_plots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_compute_progress.sizePolicy().hasHeightForWidth())
        self.plot_compute_progress.setSizePolicy(sizePolicy)
        self.plot_compute_progress.setMaximumSize(QtCore.QSize(16777215, 10))
        self.plot_compute_progress.setProperty("value", 0)
        self.plot_compute_progress.setObjectName("plot_compute_progress")
        self.verticalLayout_4.addWidget(self.plot_compute_progress)
        self.label = QtWidgets.QLabel(self.tab_plots)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_4.addWidget(self.label)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_2 = QtWidgets.QLabel(self.tab_plots)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.plot_mode = QtWidgets.QComboBox(self.tab_plots)
        self.plot_mode.setObjectName("plot_mode")
        self.plot_mode.addItem("")
        self.plot_mode.addItem("")
        self.plot_mode.addItem("")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.plot_mode)
        self.label_17 = QtWidgets.QLabel(self.tab_plots)
        self.label_17.setObjectName("label_17")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.plot_localisation_filter = QtWidgets.QComboBox(self.tab_plots)
        self.plot_localisation_filter.setObjectName("plot_localisation_filter")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.plot_localisation_filter.addItem("")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.plot_localisation_filter)
        self.label_24 = QtWidgets.QLabel(self.tab_plots)
        self.label_24.setObjectName("label_24")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_24)
        self.plot_background_subtraction_mode = QtWidgets.QComboBox(self.tab_plots)
        self.plot_background_subtraction_mode.setObjectName("plot_background_subtraction_mode")
        self.plot_background_subtraction_mode.addItem("")
        self.plot_background_subtraction_mode.addItem("")
        self.plot_background_subtraction_mode.addItem("")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.plot_background_subtraction_mode)
        self.label_26 = QtWidgets.QLabel(self.tab_plots)
        self.label_26.setObjectName("label_26")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.plot_nucleotide_filter = QtWidgets.QComboBox(self.tab_plots)
        self.plot_nucleotide_filter.setObjectName("plot_nucleotide_filter")
        self.plot_nucleotide_filter.addItem("")
        self.plot_nucleotide_filter.addItem("")
        self.plot_nucleotide_filter.addItem("")
        self.plot_nucleotide_filter.addItem("")
        self.plot_nucleotide_filter.addItem("")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.plot_nucleotide_filter)
        self.verticalLayout_4.addLayout(self.formLayout_2)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_3 = QtWidgets.QLabel(self.tab_plots)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 0, 0, 1, 1)
        self.plot_localisation_number_label = QtWidgets.QLabel(self.tab_plots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_localisation_number_label.sizePolicy().hasHeightForWidth())
        self.plot_localisation_number_label.setSizePolicy(sizePolicy)
        self.plot_localisation_number_label.setMinimumSize(QtCore.QSize(20, 0))
        self.plot_localisation_number_label.setObjectName("plot_localisation_number_label")
        self.gridLayout_4.addWidget(self.plot_localisation_number_label, 0, 2, 1, 1)
        self.plot_localisation_number = QtWidgets.QSlider(self.tab_plots)
        self.plot_localisation_number.setOrientation(QtCore.Qt.Horizontal)
        self.plot_localisation_number.setObjectName("plot_localisation_number")
        self.gridLayout_4.addWidget(self.plot_localisation_number, 0, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.tab_plots)
        self.label_18.setObjectName("label_18")
        self.gridLayout_4.addWidget(self.label_18, 1, 0, 1, 1)
        self.plot_frame_number = QtWidgets.QSlider(self.tab_plots)
        self.plot_frame_number.setOrientation(QtCore.Qt.Horizontal)
        self.plot_frame_number.setObjectName("plot_frame_number")
        self.gridLayout_4.addWidget(self.plot_frame_number, 1, 1, 1, 1)
        self.plot_frame_number_label = QtWidgets.QLabel(self.tab_plots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_frame_number_label.sizePolicy().hasHeightForWidth())
        self.plot_frame_number_label.setSizePolicy(sizePolicy)
        self.plot_frame_number_label.setMinimumSize(QtCore.QSize(20, 0))
        self.plot_frame_number_label.setObjectName("plot_frame_number_label")
        self.gridLayout_4.addWidget(self.plot_frame_number_label, 1, 2, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout_4)
        self.plot_localisation_focus = QtWidgets.QCheckBox(self.tab_plots)
        self.plot_localisation_focus.setObjectName("plot_localisation_focus")
        self.verticalLayout_4.addWidget(self.plot_localisation_focus)
        self.graph_container = QtWidgets.QWidget(self.tab_plots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graph_container.sizePolicy().hasHeightForWidth())
        self.graph_container.setSizePolicy(sizePolicy)
        self.graph_container.setMinimumSize(QtCore.QSize(0, 300))
        self.graph_container.setObjectName("graph_container")
        self.verticalLayout_4.addWidget(self.graph_container)
        spacerItem5 = QtWidgets.QSpacerItem(10, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_4.addItem(spacerItem5)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.plot_localisation_class = QtWidgets.QComboBox(self.tab_plots)
        self.plot_localisation_class.setObjectName("plot_localisation_class")
        self.plot_localisation_class.addItem("")
        self.plot_localisation_class.addItem("")
        self.plot_localisation_class.addItem("")
        self.plot_localisation_class.addItem("")
        self.plot_localisation_class.addItem("")
        self.plot_localisation_class.addItem("")
        self.plot_localisation_class.addItem("")
        self.plot_localisation_class.addItem("")
        self.plot_localisation_class.addItem("")
        self.plot_localisation_class.addItem("")
        self.gridLayout_5.addWidget(self.plot_localisation_class, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab_plots)
        self.label_5.setObjectName("label_5")
        self.gridLayout_5.addWidget(self.label_5, 0, 0, 1, 1)
        self.plot_localisation_classify = QtWidgets.QPushButton(self.tab_plots)
        self.plot_localisation_classify.setObjectName("plot_localisation_classify")
        self.gridLayout_5.addWidget(self.plot_localisation_classify, 0, 2, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.tab_plots)
        self.label_25.setObjectName("label_25")
        self.gridLayout_5.addWidget(self.label_25, 1, 0, 1, 1)
        self.plot_nucleotide_class = QtWidgets.QComboBox(self.tab_plots)
        self.plot_nucleotide_class.setObjectName("plot_nucleotide_class")
        self.plot_nucleotide_class.addItem("")
        self.plot_nucleotide_class.addItem("")
        self.plot_nucleotide_class.addItem("")
        self.plot_nucleotide_class.addItem("")
        self.gridLayout_5.addWidget(self.plot_nucleotide_class, 1, 1, 1, 1)
        self.plot_nucleotide_classify = QtWidgets.QPushButton(self.tab_plots)
        self.plot_nucleotide_classify.setObjectName("plot_nucleotide_classify")
        self.gridLayout_5.addWidget(self.plot_nucleotide_classify, 1, 2, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout_5)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem6)
        TabWidget.addTab(self.tab_plots, "")
        self.tab_IO = QtWidgets.QWidget()
        self.tab_IO.setObjectName("tab_IO")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.tab_IO)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_19 = QtWidgets.QLabel(self.tab_IO)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.verticalLayout_5.addWidget(self.label_19)
        self.gapseq_import_localisations = QtWidgets.QPushButton(self.tab_IO)
        self.gapseq_import_localisations.setObjectName("gapseq_import_localisations")
        self.verticalLayout_5.addWidget(self.gapseq_import_localisations)
        self.gapseq_import_all = QtWidgets.QPushButton(self.tab_IO)
        self.gapseq_import_all.setObjectName("gapseq_import_all")
        self.verticalLayout_5.addWidget(self.gapseq_import_all)
        self.label_23 = QtWidgets.QLabel(self.tab_IO)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.verticalLayout_5.addWidget(self.label_23)
        self.gapseq_export_at_import = QtWidgets.QCheckBox(self.tab_IO)
        self.gapseq_export_at_import.setChecked(True)
        self.gapseq_export_at_import.setObjectName("gapseq_export_at_import")
        self.verticalLayout_5.addWidget(self.gapseq_export_at_import)
        self.gapseq_export_data = QtWidgets.QPushButton(self.tab_IO)
        self.gapseq_export_data.setObjectName("gapseq_export_data")
        self.verticalLayout_5.addWidget(self.gapseq_export_data)
        self.label_20 = QtWidgets.QLabel(self.tab_IO)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.verticalLayout_5.addWidget(self.label_20)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_21 = QtWidgets.QLabel(self.tab_IO)
        self.label_21.setObjectName("label_21")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.gapseq_export_traces_filter = QtWidgets.QComboBox(self.tab_IO)
        self.gapseq_export_traces_filter.setObjectName("gapseq_export_traces_filter")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.gapseq_export_traces_filter.addItem("")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.gapseq_export_traces_filter)
        self.verticalLayout_5.addLayout(self.formLayout_3)
        self.gapseq_export_traces = QtWidgets.QPushButton(self.tab_IO)
        self.gapseq_export_traces.setObjectName("gapseq_export_traces")
        self.verticalLayout_5.addWidget(self.gapseq_export_traces)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem7)
        TabWidget.addTab(self.tab_IO, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.load_dev = QtWidgets.QPushButton(self.tab)
        self.load_dev.setObjectName("load_dev")
        self.verticalLayout.addWidget(self.load_dev)
        TabWidget.addTab(self.tab, "")

        self.retranslateUi(TabWidget)
        TabWidget.setCurrentIndex(0)
        self.localisation_channel.setCurrentIndex(1)
        self.image_import_channel.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(TabWidget)

    def retranslateUi(self, TabWidget):
        _translate = QtCore.QCoreApplication.translate
        TabWidget.setWindowTitle(_translate("TabWidget", "TabWidget"))
        self.label_4.setText(_translate("TabWidget", "Import"))
        self.localisation_channel.setItemText(0, _translate("TabWidget", "N/A"))
        self.localisation_channel.setItemText(1, _translate("TabWidget", "Left Channel (Green)"))
        self.localisation_channel.setItemText(2, _translate("TabWidget", "Right Channel (Red)"))
        self.localisation_channel.setItemText(3, _translate("TabWidget", "Brightest Channel"))
        self.localisation_import_image.setText(_translate("TabWidget", "Import Localisation Image"))
        self.label_11.setText(_translate("TabWidget", "Threshold Localisation Image"))
        self.label_7.setText(_translate("TabWidget", "Image Threshold"))
        self.localisation_threshold_label.setText(_translate("TabWidget", "200"))
        self.label_6.setText(_translate("TabWidget", "Detect Localisations"))
        self.label_27.setText(_translate("TabWidget", "Localisation Type"))
        self.localisation_type.setItemText(0, _translate("TabWidget", "Box"))
        self.localisation_type.setItemText(1, _translate("TabWidget", "Circle"))
        self.localisation_aspect_ratio_label.setText(_translate("TabWidget", "0"))
        self.localisation_area_min_label.setText(_translate("TabWidget", "0"))
        self.label_9.setText(_translate("TabWidget", "Max Localisation Area"))
        self.localisation_area_max_label.setText(_translate("TabWidget", "7"))
        self.label_8.setText(_translate("TabWidget", "Min Localisation Area"))
        self.label_28.setText(_translate("TabWidget", "Max Localisation Aspect Ratio"))
        self.localisation_detect.setText(_translate("TabWidget", "Detect Localisations"))
        self.label_12.setText(_translate("TabWidget", "Modify Bounding Boxes"))
        self.label_10.setText(_translate("TabWidget", "Bounding Box Size"))
        self.localisation_bbox_size_label.setText(_translate("TabWidget", "4"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab_localisations), _translate("TabWidget", "Localisations"))
        self.label_13.setText(_translate("TabWidget", "Import Images"))
        self.label_14.setText(_translate("TabWidget", "Image Channel"))
        self.image_import_channel.setItemText(0, _translate("TabWidget", "N/A"))
        self.image_import_channel.setItemText(1, _translate("TabWidget", "Left Channel (Green)"))
        self.image_import_channel.setItemText(2, _translate("TabWidget", "Right Channel (Red)"))
        self.image_import_channel.setItemText(3, _translate("TabWidget", "Brightest Channel"))
        self.label_15.setText(_translate("TabWidget", "Gap"))
        self.image_gap_code.setItemText(0, _translate("TabWidget", "N/A"))
        self.image_gap_code.setItemText(1, _translate("TabWidget", "A"))
        self.image_gap_code.setItemText(2, _translate("TabWidget", "T"))
        self.image_gap_code.setItemText(3, _translate("TabWidget", "C"))
        self.image_gap_code.setItemText(4, _translate("TabWidget", "G"))
        self.label_16.setText(_translate("TabWidget", "Sequence"))
        self.image_sequence_code.setItemText(0, _translate("TabWidget", "N/A"))
        self.image_sequence_code.setItemText(1, _translate("TabWidget", "A"))
        self.image_sequence_code.setItemText(2, _translate("TabWidget", "T"))
        self.image_sequence_code.setItemText(3, _translate("TabWidget", "C"))
        self.image_sequence_code.setItemText(4, _translate("TabWidget", "G"))
        self.import_image.setText(_translate("TabWidget", "Import Image"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab_images), _translate("TabWidget", "Images"))
        self.label_22.setText(_translate("TabWidget", "Compute Plot Data"))
        self.plot_compute.setText(_translate("TabWidget", "Compute"))
        self.label.setText(_translate("TabWidget", "Plot Settings"))
        self.label_2.setText(_translate("TabWidget", "Plot Mode"))
        self.plot_mode.setItemText(0, _translate("TabWidget", "Localisation Image"))
        self.plot_mode.setItemText(1, _translate("TabWidget", "Graph Image(s)"))
        self.plot_mode.setItemText(2, _translate("TabWidget", "All Images"))
        self.label_17.setText(_translate("TabWidget", "Localisation Filter"))
        self.plot_localisation_filter.setItemText(0, _translate("TabWidget", "None"))
        self.plot_localisation_filter.setItemText(1, _translate("TabWidget", "0"))
        self.plot_localisation_filter.setItemText(2, _translate("TabWidget", "1"))
        self.plot_localisation_filter.setItemText(3, _translate("TabWidget", "2"))
        self.plot_localisation_filter.setItemText(4, _translate("TabWidget", "3"))
        self.plot_localisation_filter.setItemText(5, _translate("TabWidget", "4"))
        self.plot_localisation_filter.setItemText(6, _translate("TabWidget", "5"))
        self.plot_localisation_filter.setItemText(7, _translate("TabWidget", "6"))
        self.plot_localisation_filter.setItemText(8, _translate("TabWidget", "7"))
        self.plot_localisation_filter.setItemText(9, _translate("TabWidget", "8"))
        self.plot_localisation_filter.setItemText(10, _translate("TabWidget", "9"))
        self.label_24.setText(_translate("TabWidget", "Background Subtraction mode"))
        self.plot_background_subtraction_mode.setItemText(0, _translate("TabWidget", "None"))
        self.plot_background_subtraction_mode.setItemText(1, _translate("TabWidget", "Local Background Subtraction"))
        self.plot_background_subtraction_mode.setItemText(2, _translate("TabWidget", "Global Background Subtraction"))
        self.label_26.setText(_translate("TabWidget", "Nucleotide Filter"))
        self.plot_nucleotide_filter.setItemText(0, _translate("TabWidget", "None"))
        self.plot_nucleotide_filter.setItemText(1, _translate("TabWidget", "A"))
        self.plot_nucleotide_filter.setItemText(2, _translate("TabWidget", "T"))
        self.plot_nucleotide_filter.setItemText(3, _translate("TabWidget", "C"))
        self.plot_nucleotide_filter.setItemText(4, _translate("TabWidget", "G"))
        self.label_3.setText(_translate("TabWidget", "Localisation Number"))
        self.plot_localisation_number_label.setText(_translate("TabWidget", "0"))
        self.label_18.setText(_translate("TabWidget", "Frame Number"))
        self.plot_frame_number_label.setText(_translate("TabWidget", "0"))
        self.plot_localisation_focus.setText(_translate("TabWidget", "Focus on Localisations"))
        self.plot_localisation_class.setItemText(0, _translate("TabWidget", "0"))
        self.plot_localisation_class.setItemText(1, _translate("TabWidget", "1"))
        self.plot_localisation_class.setItemText(2, _translate("TabWidget", "2"))
        self.plot_localisation_class.setItemText(3, _translate("TabWidget", "3"))
        self.plot_localisation_class.setItemText(4, _translate("TabWidget", "4"))
        self.plot_localisation_class.setItemText(5, _translate("TabWidget", "5"))
        self.plot_localisation_class.setItemText(6, _translate("TabWidget", "6"))
        self.plot_localisation_class.setItemText(7, _translate("TabWidget", "7"))
        self.plot_localisation_class.setItemText(8, _translate("TabWidget", "8"))
        self.plot_localisation_class.setItemText(9, _translate("TabWidget", "9"))
        self.label_5.setText(_translate("TabWidget", "Classify Localisation"))
        self.plot_localisation_classify.setText(_translate("TabWidget", "Classify"))
        self.label_25.setText(_translate("TabWidget", "Classify Nucleotide"))
        self.plot_nucleotide_class.setItemText(0, _translate("TabWidget", "A"))
        self.plot_nucleotide_class.setItemText(1, _translate("TabWidget", "T"))
        self.plot_nucleotide_class.setItemText(2, _translate("TabWidget", "C"))
        self.plot_nucleotide_class.setItemText(3, _translate("TabWidget", "G"))
        self.plot_nucleotide_classify.setText(_translate("TabWidget", "Classify"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab_plots), _translate("TabWidget", "Plots"))
        self.label_19.setText(_translate("TabWidget", "Import GapSeq Data"))
        self.gapseq_import_localisations.setText(_translate("TabWidget", "Import GapSeq Localisations"))
        self.gapseq_import_all.setText(_translate("TabWidget", "Import GapSeq Localisations + Images Files"))
        self.label_23.setText(_translate("TabWidget", "Export GapSeq Data"))
        self.gapseq_export_at_import.setText(_translate("TabWidget", "Export GapSeq Data in Import Directory"))
        self.gapseq_export_data.setText(_translate("TabWidget", "Export GapSeq Data"))
        self.label_20.setText(_translate("TabWidget", "Export Traces"))
        self.label_21.setText(_translate("TabWidget", "Localisation Filter"))
        self.gapseq_export_traces_filter.setItemText(0, _translate("TabWidget", "None"))
        self.gapseq_export_traces_filter.setItemText(1, _translate("TabWidget", "0"))
        self.gapseq_export_traces_filter.setItemText(2, _translate("TabWidget", "1"))
        self.gapseq_export_traces_filter.setItemText(3, _translate("TabWidget", "2"))
        self.gapseq_export_traces_filter.setItemText(4, _translate("TabWidget", "3"))
        self.gapseq_export_traces_filter.setItemText(5, _translate("TabWidget", "4"))
        self.gapseq_export_traces_filter.setItemText(6, _translate("TabWidget", "5"))
        self.gapseq_export_traces_filter.setItemText(7, _translate("TabWidget", "6"))
        self.gapseq_export_traces_filter.setItemText(8, _translate("TabWidget", "7"))
        self.gapseq_export_traces_filter.setItemText(9, _translate("TabWidget", "8"))
        self.gapseq_export_traces_filter.setItemText(10, _translate("TabWidget", "9"))
        self.gapseq_export_traces.setText(_translate("TabWidget", "Export Traces"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab_IO), _translate("TabWidget", "I/O"))
        self.load_dev.setText(_translate("TabWidget", "load DEV files"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab), _translate("TabWidget", "DEV"))
