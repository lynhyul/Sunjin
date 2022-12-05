from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt,QPoint


class Labella(QLabel):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setStyleSheet('QFrame {background-color:white;}')
        self.resize(480, 480)
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.flag = 0
        
    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QtGui.QPainter(self)
        br = QtGui.QBrush(QtGui.QColor(100, 10, 10, 40))  
        qp.setBrush(br)   
        qp.drawRect(QtCore.QRect(self.begin, self.end))

        # qp.backgroundMode
     
    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.update()
        idx1 = str(event.pos()).index('(')
        idx2 = str(event.pos()).index(')')
        self.beginpoint = str(event.pos())[idx1:idx2+1].strip()
        print("beegin = ", self.beginpoint)
    
    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()
        
    def mouseReleaseEvent(self, event):
        self.end = event.pos()
        idx1 = str(event.pos()).index('(')
        idx2 = str(event.pos()).index(')')
        self.endpoint = str(event.pos())[idx1:idx2+1].strip()
        self.update()
        print("end 1 = ", self.endpoint)    

    def pos(self) :
        return self.beginpoint,self.endpoint
    
    def flag_event(self,flag=0) :
        if flag == 1 :
            self.begin = QtCore.QPoint()
            self.end = QtCore.QPoint()
            self.update()
            self.beginpoint = '(0,0)'
            self.endpoint = '(0,0)'
            

