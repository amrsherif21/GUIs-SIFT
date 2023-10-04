#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(int(1000 / self._cam_fps))

        self.sift = cv2.xfeatures2d.SIFT_create()
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), {})

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        self.template_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        self.kp_template, self.desc_template = self.sift.detectAndCompute(self.template_img, None)

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)
        self._is_template_loaded = True

    def run_sift(self, frame):
        if not self._is_template_loaded:
            return frame  # Return original frame if template is not loaded

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = self.sift.detectAndCompute(gray_frame, None)
        matches = self.flann.knnMatch(self.desc_template, desc_frame, k=2)

        good_points = [m for m, n in matches if m.distance < 0.6 * n.distance]

        if len(good_points) > 10:
            # Finding Homography
            query_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            h, w = self.template_img.shape

            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

        else:
            # Draw matches if not a good match
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                            singlePointColor=None,
                            matchesMask=None,  # draw all matches (inliers and outliers)
                            flags=0)

            frame = cv2.drawMatchesKnn(self.template_img, self.kp_template, gray_frame, kp_frame, [good_points],
                                    frame, **draw_params)

        return frame  # Return frame with match points or homography.






    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        sift_frame = self.run_sift(frame)
        pixmap = self.convert_cv_to_pixmap(sift_frame)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())


  