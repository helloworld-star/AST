# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1285, 890)
        Form.setMinimumSize(QtCore.QSize(1285, 890))
        Form.setMaximumSize(QtCore.QSize(1285, 890))
        Form.setAutoFillBackground(True)
        Form.setStyleSheet("")
        self.Detect_Name = QtWidgets.QLabel(Form)
        self.Detect_Name.setGeometry(QtCore.QRect(190, 460, 351, 401))
        self.Detect_Name.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                       "font: 12pt \"微软雅黑\";")
        self.Detect_Name.setText("")
        self.Detect_Name.setWordWrap(True)
        self.Detect_Name.setObjectName("Detect_Name")
        self.Origin = QtWidgets.QLabel(Form)
        self.Origin.setGeometry(QtCore.QRect(580, 80, 631, 121))
        self.Origin.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Origin.setText("")
        self.Origin.setScaledContents(True)
        self.Origin.setObjectName("Origin")
        self.Layer = QtWidgets.QLabel(Form)
        self.Layer.setGeometry(QtCore.QRect(570, 340, 631, 521))
        self.Layer.setAutoFillBackground(False)
        self.Layer.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Layer.setText("")
        self.Layer.setScaledContents(True)
        self.Layer.setWordWrap(True)
        self.Layer.setObjectName("Layer")
        self.Original_Spect = QtWidgets.QLabel(Form)
        self.Original_Spect.setGeometry(QtCore.QRect(750, 20, 261, 41))
        self.Original_Spect.setStyleSheet("font: 15pt \"Adobe Heiti Std\";")
        self.Original_Spect.setObjectName("Original_Spect")
        self.Layer_map = QtWidgets.QLabel(Form)
        self.Layer_map.setGeometry(QtCore.QRect(700, 290, 371, 31))
        self.Layer_map.setStyleSheet("font: 15pt \"Adobe Heiti Std\";\n"
                                     "")
        self.Layer_map.setObjectName("Layer_map")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(250, 20, 281, 51))
        self.label_3.setStyleSheet("font: 15pt \"Adobe Heiti Std\";")
        self.label_3.setObjectName("label_3")
        self.print_filename = QtWidgets.QLabel(Form)
        self.print_filename.setGeometry(QtCore.QRect(180, 70, 361, 81))
        self.print_filename.setStyleSheet("font: 10pt \"微软雅黑\";\n"
                                          "background-color: rgb(255, 255, 255);")
        self.print_filename.setText("")
        self.print_filename.setWordWrap(True)
        self.print_filename.setObjectName("print_filename")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(260, 420, 261, 51))
        self.label.setStyleSheet("font: 15pt \"Adobe Heiti Std\";")
        self.label.setObjectName("label")
        self.logo = QtWidgets.QLabel(Form)
        self.logo.setGeometry(QtCore.QRect(20, 10, 121, 121))
        self.logo.setStyleSheet("")
        self.logo.setText("")
        self.logo.setScaledContents(True)
        self.logo.setWordWrap(True)
        self.logo.setObjectName("logo")
        self.listView = QtWidgets.QListView(Form)
        self.listView.setGeometry(QtCore.QRect(0, -50, 1301, 941))
        self.listView.setAutoFillBackground(True)
        self.listView.setStyleSheet("background-image: url(./img/Background.png);")
        self.listView.setUniformItemSizes(False)
        self.listView.setWordWrap(True)
        self.listView.setObjectName("listView")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(230, 170, 241, 241))
        self.label_2.setStyleSheet("background-image: url(./img/Icon.jfif);")
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/lzy_use.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.File = QtWidgets.QPushButton(Form)
        self.File.setGeometry(QtCore.QRect(21, 152, 120, 75))
        self.File.setStyleSheet("\n"
                                "\n"
                                "font: 75 16pt \"微软雅黑\";\n"
                                "border-radius: 10px;  \n"
                                "border: 2px groove gray;\n"
                                "\n"
                                "")
        self.File.setObjectName("File")
        self.Record = QtWidgets.QPushButton(Form)
        self.Record.setGeometry(QtCore.QRect(21, 318, 120, 75))
        self.Record.setStyleSheet("font: 75 16pt \"微软雅黑\";\n"
                                  "border-radius: 10px;  border: 2px groove gray;")
        self.Record.setObjectName("Record")
        self.Detect_2 = QtWidgets.QPushButton(Form)
        self.Detect_2.setGeometry(QtCore.QRect(21, 484, 120, 75))
        self.Detect_2.setStyleSheet("font: 75 16pt \"微软雅黑\";\n"
                                    "border-radius: 10px;  border: 2px groove gray;")
        self.Detect_2.setObjectName("Detect_2")
        self.Play = QtWidgets.QPushButton(Form)
        self.Play.setGeometry(QtCore.QRect(21, 650, 120, 75))
        self.Play.setStyleSheet("font: 75 16pt \"微软雅黑\";\n"
                                "border-radius: 10px;  border: 2px groove gray;")
        self.Play.setObjectName("Play")
        self.listView.raise_()
        self.Origin.raise_()
        self.Layer.raise_()
        self.Original_Spect.raise_()
        self.Layer_map.raise_()
        self.label_3.raise_()
        self.print_filename.raise_()
        self.label.raise_()
        self.Detect_Name.raise_()
        self.logo.raise_()
        self.File.raise_()
        self.Record.raise_()
        self.Detect_2.raise_()
        self.Play.raise_()
        self.label_2.raise_()

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "AST_GUI"))
        self.Original_Spect.setText(_translate("Form", "Original Spectrogram"))
        self.Layer_map.setText(_translate("Form", "Mean Attention Map of Layer"))
        self.label_3.setText(_translate("Form", "Current Music File"))
        self.label.setText(_translate("Form", "Detect Result"))
        self.File.setText(_translate("Form", "File"))
        self.Record.setText(_translate("Form", "Record"))
        self.Detect_2.setText(_translate("Form", "Detect"))
        self.Play.setText(_translate("Form", "Play"))
