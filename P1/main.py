from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import gui as gui
import p11

class Main(QMainWindow, gui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.Openclick)
        self.pushButton_2.clicked.connect(self.Openclick2)
        self.pushButton_3.clicked.connect(self.Openclick3)
        self.pushButton_4.clicked.connect(self.Openclick4)
    def Openclick(self):
        p11.p1()
    def Openclick2(self):
        p11.p2()
    def Openclick3(self):
        p11.p3()
    def Openclick4(self):
        p11.p4()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())