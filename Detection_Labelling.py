# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt,QPoint,QEvent
import os
import cv2
import numpy as np
from labella import Labella


class Ui_Dialog(QMainWindow):
    def trans_label_yolo(self) :
        '''
        self.sx    # 좌측 최상단 x좌표
        self.sy    # 좌측 최상단 y좌표
        self.ex - self.sx   # Box 가로
        self.ey - self.sy   # Box 세로
        '''
        dw = 1./self.w_size     ## 이미지 width
        dh = 1./self.h_size     ## 이미지 heigh
        x = (float(self.sx) + float(self.sx) + float(self.ex - self.sx))/2.0
        y = (float(self.sy) + float(self.sy) + float(self.ey - self.sy))/2.0
        w = float(self.ex - self.sx)
        h = float(self.ey - self.sy)

        self.x = round(x*dw, 6)
        self.w = round(w*dw, 6)
        self.y = round(y*dh, 6) # 6자리 표시
        self.h = round(h*dh, 6)

    
    def __init__(self) :
        super().__init__()
        self.root_path = ''
        self.clss = -1
        self.h_size = 480
        self.w_size = 480
        self.beginpoint = None
        self.endpoint = None
        self.file_name = None
        
    
    def create_folder(self,directory):
    # 폴더 생성 함수
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            self.textWidget.appendPlainText('Error: Creating directory. ' + directory)
    
    
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1096, 845)
        cnt = 40
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10+cnt, 1061, 781))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")

        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(910, cnt-20, 121, 31))
        self.pushButton.setObjectName("load1")
        self.pushButton.clicked.connect(self.load_clicked)
        
        self.pushButton_6 = QtWidgets.QPushButton(self.tab)
        self.pushButton_6.setGeometry(QtCore.QRect(910, 30+cnt, 121, 31))
        self.pushButton_6.setObjectName("clear")
        self.pushButton_6.clicked.connect(self.box_clear)
        
        self.pushButton_5 = QtWidgets.QPushButton(self.tab)
        self.pushButton_5.setGeometry(QtCore.QRect(910, 80+cnt, 121, 31))
        self.pushButton_5.setObjectName("next_label")
        self.pushButton_5.clicked.connect(self.rect_draw)
        
        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setGeometry(QtCore.QRect(910, 130+cnt, 121, 31))
        self.pushButton_2.setObjectName("save1")
        self.pushButton_2.clicked.connect(self.save_clicked)
        
        self.pushButton_3 = QtWidgets.QPushButton(self.tab)
        self.pushButton_3.setGeometry(QtCore.QRect(910, 180+cnt, 121, 31))
        self.pushButton_3.setObjectName("next1")
        self.pushButton_3.clicked.connect(self.next_clicked)
        
        self.pushButton_4 = QtWidgets.QPushButton(self.tab)
        self.pushButton_4.setGeometry(QtCore.QRect(910, 230+cnt, 121, 31))
        self.pushButton_4.setObjectName("back1")
        self.pushButton_4.clicked.connect(self.back_clicked)
        self.label = Labella(self.tab)
        self.label.setGeometry(QtCore.QRect(30, 50, 561, 511))
        self.label.setObjectName("label")
        self.pushButton_9 = QtWidgets.QPushButton(self.tab)
        self.pushButton_9.setGeometry(QtCore.QRect(910, 280+cnt, 121, 31))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.clicked.connect(self.clist_load)
        
        
        self.label2 = QtWidgets.QLabel(self.tab)
        self.label2.setGeometry(QtCore.QRect(915, 370, 121, 30))
        self.label2.setObjectName("label2")
        
        self.label2.setFont(QtGui.QFont("궁서",10)) #폰트,크기 조절
        
        self.listWidget = QtWidgets.QListWidget(self.tab)
        self.listWidget.setGeometry(QtCore.QRect(910, 400, 121, 330))
        self.listWidget.setObjectName("listWidget")
        self.listWidget.itemClicked.connect(self.list_clicked)
        
        self.label3 = QtWidgets.QLabel(self.tab)
        self.label3.setGeometry(QtCore.QRect(710, 15, 200, 25))
        self.label3.setObjectName("label3")
        
        self.label3.setFont(QtGui.QFont("궁서",10)) #폰트,크기 조절
        
        self.label4 = QtWidgets.QLabel(self.tab)
        self.label4.setGeometry(QtCore.QRect(650, 45, 200, 25))
        self.label4.setObjectName("label4")
        
        self.label4.setFont(QtGui.QFont("바탕",8)) #폰트,크기 조절
        
        self.listWidget2 = QtWidgets.QListWidget(self.tab)
        self.listWidget2.setGeometry(QtCore.QRect(620, 70, 250, 660))
        self.listWidget2.setObjectName("listWidget")
        self.listWidget2.doubleClicked.connect(self.file_list_fun)
        
        self.textWidget = QPlainTextEdit(self.tab)
        self.textWidget.setGeometry(QtCore.QRect(30, 550, 480, 180))
        # self.textWidget.setPlainText('log')
        
        
        self.t2_pushButton = QtWidgets.QPushButton(self.tab_2)
        self.t2_pushButton.setGeometry(QtCore.QRect(910, cnt-20, 121, 31))
        self.t2_pushButton.setObjectName("load1")
        self.t2_pushButton.clicked.connect(self.load_clicked)
        
        self.t2_pushButton_6 = QtWidgets.QPushButton(self.tab_2)
        self.t2_pushButton_6.setGeometry(QtCore.QRect(910, 30+cnt, 121, 31))
        self.t2_pushButton_6.setObjectName("clear")
        self.t2_pushButton_6.clicked.connect(self.box_clear)
        
        self.t2_pushButton_5 = QtWidgets.QPushButton(self.tab_2)
        self.t2_pushButton_5.setGeometry(QtCore.QRect(910, 80+cnt, 121, 31))
        self.t2_pushButton_5.setObjectName("next_label")
        self.t2_pushButton_5.clicked.connect(self.rect_draw)
        
        self.t2_pushButton_2 = QtWidgets.QPushButton(self.tab_2)
        self.t2_pushButton_2.setGeometry(QtCore.QRect(910, 130+cnt, 121, 31))
        self.t2_pushButton_2.setObjectName("save1")
        self.t2_pushButton_2.clicked.connect(self.save_clicked)
        
        self.t2_pushButton_3 = QtWidgets.QPushButton(self.tab_2)
        self.t2_pushButton_3.setGeometry(QtCore.QRect(910, 180+cnt, 121, 31))
        self.t2_pushButton_3.setObjectName("next1")
        self.t2_pushButton_3.clicked.connect(self.next_clicked)
        
        self.t2_pushButton_4 = QtWidgets.QPushButton(self.tab_2)
        self.t2_pushButton_4.setGeometry(QtCore.QRect(910, 230+cnt, 121, 31))
        self.t2_pushButton_4.setObjectName("back1")
        self.t2_pushButton_4.clicked.connect(self.back_clicked)
        self.t2_label = Labella(self.tab_2)
        self.t2_label.setGeometry(QtCore.QRect(30, 50, 561, 511))
        self.t2_label.setObjectName("label")
        self.t2_pushButton_9 = QtWidgets.QPushButton(self.tab_2)
        self.t2_pushButton_9.setGeometry(QtCore.QRect(910, 280+cnt, 121, 31))
        self.t2_pushButton_9.setObjectName("t2_pushButton_9")
        self.t2_pushButton_9.clicked.connect(self.clist_load)
        
        
        self.t2_label2 = QtWidgets.QLabel(self.tab_2)
        self.t2_label2.setGeometry(QtCore.QRect(915, 370, 121, 30))
        self.t2_label2.setObjectName("label2")
        
        self.t2_label2.setFont(QtGui.QFont("궁서",10)) #폰트,크기 조절
        
        self.t2_listWidget = QtWidgets.QListWidget(self.tab_2)
        self.t2_listWidget.setGeometry(QtCore.QRect(910, 400, 121, 330))
        self.t2_listWidget.setObjectName("t2_listWidget")
        self.t2_listWidget.itemClicked.connect(self.list_clicked)
        
        self.t2_label3 = QtWidgets.QLabel(self.tab_2)
        self.t2_label3.setGeometry(QtCore.QRect(710, 15, 200, 25))
        self.t2_label3.setObjectName("t2_label3")
        
        self.t2_label3.setFont(QtGui.QFont("궁서",10)) #폰트,크기 조절
        
        self.t2_label4 = QtWidgets.QLabel(self.tab_2)
        self.t2_label4.setGeometry(QtCore.QRect(650, 45, 200, 25))
        self.t2_label4.setObjectName("label4")
        
        self.t2_label4.setFont(QtGui.QFont("바탕",8)) #폰트,크기 조절
        
        self.t2_listWidget2 = QtWidgets.QListWidget(self.tab_2)
        self.t2_listWidget2.setGeometry(QtCore.QRect(620, 70, 250, 660))
        self.t2_listWidget2.setObjectName("listWidget")
        self.t2_listWidget2.doubleClicked.connect(self.file_list_fun)
        
        self.t2_textWidget = QPlainTextEdit(self.tab_2)
        self.t2_textWidget.setGeometry(QtCore.QRect(30, 550, 480, 180))
        # self.textWidget.setPlainText('log')
        
        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Load"))
        self.pushButton_2.setText(_translate("Dialog", "Save"))
        self.pushButton_3.setText(_translate("Dialog", "Next"))
        self.pushButton_4.setText(_translate("Dialog", "Back"))
        self.pushButton_5.setText(_translate("Dialog", "Next_label"))
        self.pushButton_9.setText(_translate("Dialog", "Class_dir"))
        self.pushButton_6.setText(_translate("Dialog", "Box_Clear"))
        self.label3.setText('File List')
        self.label2.setText('Select Labell')
        self.label.setText("")
        self.label4.setText(f'File Name :     {self.file_name}'   )
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "Tab 2"))
        
        self.t2_pushButton.setText(_translate("Dialog", "Load"))
        self.t2_pushButton_2.setText(_translate("Dialog", "Save"))
        self.t2_pushButton_3.setText(_translate("Dialog", "Next"))
        self.t2_pushButton_4.setText(_translate("Dialog", "Back"))
        self.t2_pushButton_5.setText(_translate("Dialog", "Next_label"))
        self.t2_pushButton_9.setText(_translate("Dialog", "Class_dir"))
        self.t2_pushButton_6.setText(_translate("Dialog", "Box_Clear"))
        self.t2_label3.setText('File List')
        self.t2_label2.setText('Select Labell')
        self.t2_label.setText("")
        self.t2_label4.setText(f'File Name :     {self.file_name}'   )

        
    def box_clear(self) :
        self.points = []
        self.label.flag_event(flag=1)
        self.image_load_fun(self.fname[0])
        
     
    def file_list_fun(self) :
        self.fname = self.listWidget2.currentItem().text()   ## 이미지 파일 이름 얻기
        if self.fname[-3:] == 'jpg' or self.fname[-3:] == 'png' :
            img_path = f'{self.root_path}/{self.fname}' 
            self.image_load_fun(img_path)
            self.label4.setText(f'File Name :     {self.fname}'   )
        else :
            self.textWidget.appendPlainText("이미지 파일을 선택해주세요")
            
    def rect_draw(self) :
        if self.clss >= 0 :
            self.beginpoint,self.endpoint = self.label.pos()
            # self.label2.repaint()
            self.beginpoint = self.beginpoint.replace('(','')
            self.beginpoint= self.beginpoint.replace(')','')
            self.endpoint = self.endpoint.replace('(','')
            self.endpoint= self.endpoint.replace(')','')
            self.sx = int(self.beginpoint.split(',')[0])
            self.sy = int(self.beginpoint.split(',')[-1])
            self.ex = int(self.endpoint.split(',')[0])
            self.ey = int(self.endpoint.split(',')[-1])
            img = cv2.rectangle(self.img,(self.sx,self.sy),(self.ex,self.ey),(255,0,0))
            h,w,c, = img.shape
            qImg=QtGui.QImage(img.data,w,h,w*c,QtGui.QImage.Format_RGB888)
            pixmap=QtGui.QPixmap.fromImage(qImg)
            self.label.setPixmap(pixmap)
            self.trans_label_yolo()
            self.points.append(f'{self.clss} {self.x} {self.y} {self.w} {self.h}\n')
        else :
            self.textWidget.appendPlainText("Class 정보를 입력해주세요")

    
    def image_load_fun(self,img_path) :
        self.points = []
        self.flag = 0
        self.src_img = cv2.imread(img_path)
        self.img = cv2.cvtColor(self.src_img,cv2.COLOR_RGB2BGR)
        self.img = cv2.resize(self.img,(self.h_size,self.w_size))
        h,w,c, = self.img.shape
        qImg=QtGui.QImage(self.img.data,w,h,w*c,QtGui.QImage.Format_RGB888)

        pixmap=QtGui.QPixmap.fromImage(qImg)

        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(),pixmap.height())
 
        
    def load_clicked(self) :
        self.fname = QFileDialog.getOpenFileName()   ## 이미지 파일 이름 얻기
        if self.fname[0] :
            self.image_load_fun(self.fname[0])
            self.file_name = self.fname[0].split('/')[-1]
            self.root_path = self.fname[0].rstrip(self.file_name)  ## root_path를 가져온다
            self.flist = os.listdir(self.root_path)
            self.idx = self.flist.index(self.file_name)
            for idx,item in enumerate(self.flist) :
                self.listWidget2.insertItem(idx,item)
            self.label4.setText(f'File Name :     {self.file_name}'   )
        else:
            self.textWidget.appendPlainText("먼저, 파일을 불러오세요")
           
            
    def clist_load(self) :
        fname = QFileDialog.getExistingDirectory()   ## 라벨 폴더 경로 얻기
        self.listWidget.clear()
        if fname :
            self.clist = os.listdir(fname)
        for idx,item in enumerate(self.clist) :
            self.listWidget.insertItem(idx,item)
            
    
    def list_clicked(self) :
        self.clss = self.listWidget.currentRow()
        
        
    def next_clicked(self) :
        if self.idx == len(self.flist) -1 :
            self.textWidget.appendPlainText("마지막 파일입니다.")
        while self.idx < len(self.flist)-1 :
            self.idx += 1
            if self.flist[self.idx][-3:] == 'jpg' or self.flist[self.idx][-3:] == 'png' :
                img_path = f'{self.root_path}/{self.flist[self.idx]}'
                self.image_load_fun(img_path)
                self.label4.setText(f'File Name :     {self.flist[self.idx]}'   )
                self.file_name = self.flist[self.idx]
                break
        
                    
    def back_clicked(self) :
        if self.idx == 0:
            self.textWidget.appendPlainText("1번째 파일입니다.")
        while self.idx > 0  :
            self.idx -= 1
            if self.idx >= 0 :
                if self.flist[self.idx][-3:] == 'jpg' or self.flist[self.idx][-3:] == 'png' :
                    img_path = f'{self.root_path}/{self.flist[self.idx]}'
                    self.image_load_fun(img_path)
                    self.label4.setText(f'File Name :     {self.flist[self.idx]}'   )
                    self.file_name = self.flist[self.idx]
                    break
                
    def keyPressEvent(self, e):
        print("key")
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_F:
            self.showFullScreen()
        elif e.modifiers() & Qt.ControlModifier:
            if e.key() == Qt.Key_S:
                self.save_clicked()
        
    
    def save_clicked(self) :
        self.flag = 1
        if self.clss >= 0 :
            self.create_folder(self.root_path+f'/train/')
            self.create_folder(self.root_path+f'/label/')
            
            ## 원본 이미지 저장
            cv2.imwrite(self.root_path+f'/train/{self.file_name}',self.src_img)
            self.textWidget.appendPlainText(f'image saved : {self.file_name}')
            
            ## label 정보 저장
            label = open(f"{self.root_path}/label/{self.file_name[:-4]}.txt", 'w')
            try :
                self.beginpoint,self.endpoint = self.label.pos()
                # self.label2.repaint()
                self.beginpoint = self.beginpoint.replace('(','')
                self.beginpoint= self.beginpoint.replace(')','')
                self.endpoint = self.endpoint.replace('(','')
                self.endpoint= self.endpoint.replace(')','')
                self.sx = int(self.beginpoint.split(',')[0])
                self.sy = int(self.beginpoint.split(',')[-1])
                self.ex = int(self.endpoint.split(',')[0])
                self.ey = int(self.endpoint.split(',')[-1])
                self.points.append(f'{self.clss} {self.x} {self.y} {self.w} {self.h}')
                self.trans_label_yolo()
                for point in self.points :
                    label.write(point)
                    self.textWidget.appendPlainText(f'label saved : {point}')
                # label.write(f'{self.clss} {self.x} {self.y} {self.w} {self.h}')
                
            except :
                label.write('')
            label.close()
            self.label.flag_event(flag=1)
            self.next_clicked()
            # self.textWidget.appendPlainText(self.clss,self.x, self.y, self.w, self.h)
        else :
            self.textWidget.appendPlainText("Class 정보를 입력해주세요")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
