from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import gui as gui
import P4

class Main(QMainWindow, gui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)
        self.transformation.clicked.connect(self.Openclick)
        self.Rotation.textChanged.connect(self.save)
        self.Scaling.textChanged.connect(self.save)
        self.Tx.textChanged.connect(self.save)
        self.Ty.textChanged.connect(self.save)
        self.display.clicked.connect(self.displays)

    def Openclick(self):
        P4.open()
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