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
        self.groupBox.setGeometry(QtCore.QRect(220, 130, 451, 331))
        self.groupBox.setObjectName("groupBox")
        self.Rotation = QtWidgets.QLineEdit(self.groupBox)
        self.Rotation.setGeometry(QtCore.QRect(160, 50, 113, 22))
        self.Rotation.setObjectName("Rotation")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(90, 50, 58, 15))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(90, 110, 58, 15))
        self.label_2.setObjectName("label_2")
        self.Scaling = QtWidgets.QLineEdit(self.groupBox)
        self.Scaling.setGeometry(QtCore.QRect(160, 100, 113, 22))
        self.Scaling.setObjectName("Scaling")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(90, 160, 58, 15))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(90, 210, 58, 15))
        self.label_4.setObjectName("label_4")
        self.Tx = QtWidgets.QLineEdit(self.groupBox)
        self.Tx.setGeometry(QtCore.QRect(160, 150, 113, 22))
        self.Tx.setObjectName("Tx")
        self.Ty = QtWidgets.QLineEdit(self.groupBox)
        self.Ty.setGeometry(QtCore.QRect(160, 200, 113, 22))
        self.Ty.setObjectName("Ty")
        self.transformation = QtWidgets.QPushButton(self.groupBox)
        self.transformation.setGeometry(QtCore.QRect(110, 260, 181, 28))
        self.transformation.setObjectName("transformation")
        self.display = QtWidgets.QPushButton(self.groupBox)
        self.display.setGeometry(QtCore.QRect(330, 130, 71, 51))
        self.display.setObjectName("display")
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
        self.groupBox.setTitle(_translate("MainWindow", "4. Transformation"))
        self.label.setText(_translate("MainWindow", "Rotation:"))
        self.label_2.setText(_translate("MainWindow", "Scaling:"))
        self.label_3.setText(_translate("MainWindow", "Tx:"))
        self.label_4.setText(_translate("MainWindow", "Ty:"))
        self.transformation.setText(_translate("MainWindow", "4. Transformation"))
        self.display.setText(_translate("MainWindow", "Display"))