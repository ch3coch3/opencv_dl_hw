from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import gui as gui
import main

class Main(QMainWindow, gui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.Openclick)
        self.pushButton_2.clicked.connect(self.Openclick1)
        self.pushButton_3.clicked.connect(self.Openclick2)
        self.pushButton_4.clicked.connect(self.Openclick3)
        self.pushButton_5.clicked.connect(self.Openclick4)

    def Openclick(self):
        main.showTrainImage()
    def Openclick1(self):
        main.showParameter()
    def Openclick2(self):
        main.showmodel()
    def Openclick3(self):
        main.showAcc()
    def Openclick4(self):
        pass

    def save(self):
        global angle,scale,tx,ty
        angle = self.Rotation.text()
        scale = self.Scaling.text()
        tx = self.Tx.text()
        ty = self.Ty.text()
    def displays(self):
        P4.p(angle,scale,tx,ty)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())