
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton,  QPlainTextEdit, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, QCoreApplication
from PySide2.QtGui import QPixmap

import time
import os
import sys
import cv2

from PIL import Image
from ui.fun import cfg, inference

class Ui():
    def __init__(self):

        # 从文件中加载UI定义
        qfile = QFile("segmentation.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()

        self.ui = QUiLoader().load(qfile)

        # slot处理signal
        self.ui.pb_select.clicked.connect(self.open_image)
        self.ui.pb_run.clicked.connect(self.run)
        self.ui.pb_clear.clicked.connect(self.clear)
        self.ui.pb_exit.clicked.connect(self.exit)
        self.ui.pb_save.clicked.connect(self.save)

    def open_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self.ui, "选择图片", "", "*.png;;All Files(*)")    # "*.jpg;;*.png;;All Files(*)"
        self.fname = str(imgName)
        self.show_open_path()
        jpg = QPixmap(imgName).scaled(self.ui.label_img.width(), self.ui.label_img.height())
        self.ui.label_img.setPixmap(jpg)
        self.start_time = time.time()

    def show_open_path(self):
        self.ui.tb_path_open.setText(self.fname)

    def show_image(self, img):
        # PIL格式转QPixmap格式
        img = Image.fromarray(img)
        from PIL import ImageQt
        pixmap = ImageQt.toqpixmap(img)
        pixmap = pixmap.scaled(self.ui.label_img.width(), self.ui.label_img.height())
        self.ui.label_result.setPixmap(pixmap)

    def show_pixnum(self):

        self.ui.tb_pixnum.setText("{} 个   {} %".format(self.pix_num, "%.2f" % (self.pix_num/(256*256))))

    def show_time(self):
        self.ui.tb_time.setText("{} s".format("%.2f" % (self.end_time)))

    def run(self):
        print("file_name", self.fname)
        # ============================test============================================================================
        self.prediction, self.pix_num = inference(cfg, self.fname)
        self.show_image(self.prediction)
        # =============================================================================================================
        self.show_pixnum()
        self.end_time = time.time() - self.start_time
        self.show_time()

    def save(self):
        sp = self.fname.split(".")
        self.save_path = sp[0] + "_result." + sp[1]
        self.ui.tb_path_save.setText(self.save_path)
        cv2.imwrite(self.save_path, self.prediction)

    def clear(self):
        self.ui.tb_path_open.setText("")
        self.ui.label_img.setText("                       原图显示区域")
        self.ui.label_result.setText("                     结果显示区域")
        self.ui.tb_pixnum.setText("")
        self.ui.tb_time.setText("")
        self.ui.tb_path_save.setText("")



    def exit(self):
        self.ui.close()



if __name__ == '__main__':
    app = QApplication(sys.argv)    # 提供了整个图形界面程序的底层管理功能
    # app = QApplication([])  # 提供了整个图形界面程序的底层管理功能
    ex = Ui()
    ex.ui.show()    # 显示主窗口
    # app.exec_()
    sys.exit(app.exec_())    #进入QApplication的事件处理循环，接收用户的输入事件（），并且分配给相应的对象去处理
