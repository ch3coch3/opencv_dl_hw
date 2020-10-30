from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import gui as gui
import P2

class Main(QMainWindow, gui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)
        self.median_filter_2.clicked.connect(self.Openclick)
        self.gaussian_filter.clicked.connect(self.Openclick2)
        self.bilateral_filter.clicked.connect(self.Openclick3)
    def Openclick(self):
        P2.p1()
    def Openclick2(self):
        P2.p2()
    def Openclick3(self):
        P2.p3()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())