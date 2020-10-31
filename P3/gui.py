# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(150, 110, 421, 331))
        self.groupBox.setObjectName("groupBox")
        self.gaussian_blur = QtWidgets.QPushButton(self.groupBox)
        self.gaussian_blur.setGeometry(QtCore.QRect(80, 70, 211, 28))
        self.gaussian_blur.setObjectName("gaussian_blur")
        self.SobelX = QtWidgets.QPushButton(self.groupBox)
        self.SobelX.setGeometry(QtCore.QRect(80, 130, 211, 28))
        self.SobelX.setObjectName("SobelX")
        self.SobelY = QtWidgets.QPushButton(self.groupBox)
        self.SobelY.setGeometry(QtCore.QRect(80, 190, 211, 28))
        self.SobelY.setObjectName("SobelY")
        self.magnitude = QtWidgets.QPushButton(self.groupBox)
        self.magnitude.setGeometry(QtCore.QRect(80, 240, 211, 28))
        self.magnitude.setObjectName("magnitude")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "3. Edge Detection"))
        self.gaussian_blur.setText(_translate("MainWindow", "3.1 Gaussian Blur"))
        self.SobelX.setText(_translate("MainWindow", "3.2 Sobel X"))
        self.SobelY.setText(_translate("MainWindow", "3.3 Sobel Y"))
        self.magnitude.setText(_translate("MainWindow", "3.4 Magnitude"))
