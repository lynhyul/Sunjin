# GUI
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QProgressBar, QSlider, QListView, QTextEdit,QListWidget
from PyQt5.QtGui import QIcon, QFont, QStandardItemModel, QStandardItem
from PyQt5.QtCore import QProcess, Qt,QRect
from qt_material import apply_stylesheet

# 경로
from glob import glob
import os

class main(QWidget):
    def __init__(self):
        super().__init__()
        self.UI_init()
        
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.handle_finished)
        self.bar_value = 0
        
    def create_folder(self,directory):
    # 폴더 생성 함수
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("create error")
            pass

        
    def UI_init(self) :
        # 가상환경 실행
        global venv_path
        venv_path = 'C:/PYQT5/pyqt5/Scripts/activate'
        # 오픈경로 파일 탐색기 버튼
        fileopen_text = ""
        self.fileopen_button = QPushButton('XARY 이미지 저장 경로',self)
        self.fileopen_button.clicked.connect(self.fileopen)
        self.fileopen_button.move(10,10)
        self.fileopen_button.resize(200,50)
        # 오픈 경로 파일 라벨
        self.fileopen_label = QLabel('저장경로', self)
        self.fileopen_label.move(220,10)
        self.fileopen_label.resize(500,50)
        # 저장경로 파일 탐색기 버튼
        self.filesave_button = QPushButton('분석 결과 저장 경로',self)
        self.filesave_button.clicked.connect(self.filesave)
        self.filesave_button.move(10,70)
        self.filesave_button.resize(200,50)
        # 저장 경로 파일 라벨
        self.filesave_label = QLabel('저장경로', self)
        self.filesave_label.move(220,70)
        self.filesave_label.resize(500,50)
        # 저장 폴더 명 라벨
        self.filefolder_label = QPushButton('저장 폴더 명', self)
        self.filefolder_label.move(390,70)
        self.filefolder_label.resize(150,50)
        # 저장 폴더 명 입력
        self.filefolder_Edit = QTextEdit(self)
        self.filefolder_Edit.move(550,70)
        self.filefolder_Edit.resize(300,50)
        self.filefolder_Edit.setAcceptRichText(False)
        
        # 프로그래스 바
        self.bar = QProgressBar(self)
        self.bar.setGeometry(150,180,700,50)
        
        # confidence 슬라이더 라벨
        self.confidence_slider_label = QLabel('민감도(Confidence)', self)
        self.confidence_slider_label.move(15,120)
        self.confidence_slider_label.resize(200,50)
        self.confidence_slider_label.setFont(QFont("Arial", 50, QFont.Bold))

        # confidence 라벨 레이아웃 추가
        #layout = QHBoxLayout()
        #layout.addWidget(self.confidence_slider_label)

        # confidence 슬라이더
        self.confidence_slider = QSlider(Qt.Horizontal, self)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(5)
        self.confidence_slider.setValue(20)
        self.confidence_slider.move(180,135)
        self.confidence_slider.resize(670,50)

        # 슬라이더 값을 표시하는 라벨
        self.confidence_result_label = QLabel(str(self.confidence_slider.value()),self)
        self.confidence_result_label.move(870,135)
        self.confidence_result_label.resize(40,20)
        #layout.addWidget(self.confidence_result_label)

        # 슬라이더 값을 변경할 때마다 호출하는 함수 생성
        def update_confindece(value) :
            self.confidence_result_label.setText(str(value))
        self.confidence_slider.valueChanged.connect(update_confindece)

        # confidence 슬라이더 레이아웃 추가
        #layout.addWidget(self.confidence_slider)

        # YOLO 돌리기
        self.run_button = QPushButton('RUN',self)
        self.run_button.clicked.connect(self.run)
        self.run_button.move(10,180)
        self.run_button.resize(100,50)
        
        ## ListView
        
        
        # self.view = QListView(self)
        self.listWidget = QListWidget(self)
        self.listWidget.setGeometry(QRect(0, 250, 600, 900))
        self.listWidget.setObjectName("listWidget")
        self.listWidget.itemClicked.connect(self.list_clicked)
        # self.view_model = QStandardItemModel()
        # self.view.move(10,250)
        # self.view.resize(600,900)
        
        self.toolTip() 
        #self.setLayout(layout)
        self.setWindowTitle('AXSS(AI XRAY Search system for Sunjin)')
        self.setWindowIcon(QIcon('C:\PYQT5\XSS\sunjin.jpg'))
        self.setGeometry(0,0,1500,1200)
        self.show()

    def fileopen(self) :
        # 파일 명 불러오기
        global filename
        filename = QFileDialog.getOpenFileName(self, 'Open File')
        filename = filename[0].split("/")
        filename_temp = ""
        for _ in range(len(filename)-1) :
            filename_temp += filename[_] + "/"
        fileopen = filename_temp
        
        # 파일 명 띄우기
        self.fileopen_label.setText(fileopen)
        # 글로벌 변수로 빼줘야함
        global fileopen_text
        fileopen_text = fileopen
        fileopen_len = len(glob(str(fileopen_text) + "*"))
        self.bar.setRange(0,fileopen_len)
        
    def filesave(self) :
        # 파일 명 불러오기
        global filename
        filename = QFileDialog.getOpenFileName(self, 'Save File')
        filename = filename[0].split("/")
        filename_temp = ""
        for _ in range(len(filename)-1) :
            filename_temp += filename[_] + "/"
        filesave = filename_temp
        
        # 파일 명 띄우기
        self.filesave_label.setText(filesave)
        # 글로벌 변수로 빼줘야함
        global filesave_text
        filesave_text = filesave
        
    def run(self) :
        self.bar_value = 0
        global fileopen_text, filesave_text
        detect_py_dir = "C:\PYQT5\yolov7\detect.py"
            
        confidence_level = str(float( self.confidence_result_label.text())*0.01)
        weights = glob('C:/PYQT5/yolov7/test/*')
        weights_text = ""
        for _ in weights :
            weights_text += _ + " "
        
        name_text = self.filefolder_Edit.toPlainText() 
        
        # cmd와 연결
        run_command = f'cmd /c {venv_path} && python ' + detect_py_dir + ' --name ' + name_text +' --weights ' + weights_text + ' --conf ' + confidence_level + ' --img-size 1280' + ' --source ' + fileopen_text + ' --project ' + filesave_text + ' --save-txt --no-trace'

        # QProcess 실행
        self.process.start(run_command)
        
        # 신호 걸기
        run_result_list = glob(self.filefolder_Edit.toPlainText() + "/*")
        for item in run_result_list :
            item.replace(".txt","")
            self.view_model.appendRow(QStandardItem(item))
        self.view.setModel(self.view_model)

    # QProcess의 readyReadStandardOutput 시그널을 처리하는 슬롯
    def handle_stdout(self):
        output = self.process.readAllStandardOutput().data().decode('utf-8')
        print(output)
        self.bar_value += output.count("Inference")
        self.bar.setValue(self.bar_value)
        print(self.bar_value)
        
    def list_clicked(self) :
        self.clss = self.listWidget.currentRow()
        

    # QProcess의 readyReadStandardError 시그널을 처리하는 슬롯
    def handle_stderr(self):
        error = self.process.readAllStandardError().data().decode('utf-8')
        print(error)
        
    # QProcess의 finished 시그널을 처리하는 슬롯
    def handle_finished(self, exit_code, exit_status):
        print(f'Finished with exit code {exit_code}, exit status {exit_status}')
        
    def toolTip(self) :
        self.fileopen_button.setToolTip('한글 경로는 지원되지 않습니다.')
        self.filesave_button.setToolTip('한글 경로는 지원되지 않습니다.')

  
loop = QApplication(sys.argv)
# 시트 설정
apply_stylesheet(loop, theme='dark_teal.xml')
instance = main()
loop.exec_()