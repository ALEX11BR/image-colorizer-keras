#!/usr/bin/env python3

import sys

import numpy as np

from tensorflow.keras.models import load_model

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QFrame, QPushButton, QFileDialog

from common import load_image_for_model

def np_to_qpixmap(np_array):
    np_array = (np_array * 256).astype(np.uint8)
    if len(np_array.shape) == 2:
        channels = 1
        height, width = np_array.shape
    else:
        height, width, channels = np_array.shape
    bytes_per_line = channels * width
    image_format = QImage.Format_Grayscale8 if channels == 1 else QImage.Format_RGB888

    q_image = QImage(np_array.data, width, height, bytes_per_line, image_format)
    return QPixmap.fromImage(q_image)
class AppScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image colorizer")
        self.setGeometry(100, 100, 600, 800)

        self.root_layout = QVBoxLayout(self)

        self.selected_model_label = QLabel("No selected model")
        self.root_layout.addWidget(self.selected_model_label)

        self.select_model_button = QPushButton("Select model")
        self.select_model_button.clicked.connect(self.select_model)
        self.root_layout.addWidget(self.select_model_button)

        self.original_image_frame = QFrame()
        self.original_image_frame.setFrameShape(QFrame.StyledPanel)
        self.original_image_frame.setFrameShadow(QFrame.Raised)
        self.root_layout.addWidget(self.original_image_frame)

        self.original_image = QLabel(self.original_image_frame)
        self.original_image.setText("No selected image")
        self.original_image.setMinimumSize(200, 200)

        self.select_image_button = QPushButton("Select image")
        self.select_image_button.clicked.connect(self.select_image)
        self.root_layout.addWidget(self.select_image_button)

        self.colorized_image_frame = QFrame()
        self.colorized_image_frame.setFrameShape(QFrame.StyledPanel)
        self.colorized_image_frame.setFrameShadow(QFrame.Raised)
        self.root_layout.addWidget(self.colorized_image_frame)

        self.colorized_image = QLabel(self.colorized_image_frame)
        self.colorized_image.setText("Image not colorized yet")
        self.colorized_image.setMinimumSize(200, 200)

        self.save_image_button = QPushButton("Save image")
        self.save_image_button.clicked.connect(self.save_image)
        self.root_layout.addWidget(self.save_image_button)

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose model file", "")
        if file_path != "":
            QApplication.setOverrideCursor(Qt.WaitCursor)

            self.ml_model = load_model(file_path)
            self.selected_model_label.setText("Selected model from " + file_path)

            QApplication.restoreOverrideCursor()

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose image", "")
        if file_path != "":
            QApplication.setOverrideCursor(Qt.WaitCursor)

            image_data = load_image_for_model(file_path)
            self.original_image.setPixmap(np_to_qpixmap(image_data))

            colorized_image = self.ml_model.predict(np.array([image_data]))[0]
            self.colorized_pixmap = np_to_qpixmap(colorized_image)
            self.colorized_image.setPixmap(self.colorized_pixmap)

            QApplication.restoreOverrideCursor()

    def save_image(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Select file to save to", "")
        if file_name != "":
            QApplication.setOverrideCursor(Qt.WaitCursor)

            self.colorized_pixmap.save(file_name)

            QApplication.restoreOverrideCursor()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_screen = AppScreen()
    app_screen.show()
    sys.exit(app.exec_())
