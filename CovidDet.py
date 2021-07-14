import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from user_interface.interface import Ui_MainWindow

def script_method(fn, _rcb=None):
    return fn

def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj

import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.setWindowTitle("COVID-19肺部CT辅助诊断系统v1.0")
    mainWindow.show()
    sys.exit(app.exec_())