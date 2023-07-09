import sys
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication

class ImageProcessingThread(QThread):
    imageProcessed = pyqtSignal(str)

    def __init__(self, folder):
        QThread.__init__(self)
        self.folder = folder

    def run(self):
        while True:
            # 폴더를 감시하고 새로운 이미지 파일이 생성되면 실행
            for filename in os.listdir(self.folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(self.folder, filename)
                    # 딥러닝 모델 실행 (여기에 모델 예측 코드 추가)
                    print("처리 중인 이미지:", image_path)
                    self.imageProcessed.emit(image_path)
                    time.sleep(1)  # 예시로 1초 동안 대기

            time.sleep(1)  # 1초마다 폴더를 확인하기 위한 대기 시간

class ImageCreationThread(QThread):
    imageCreated = pyqtSignal(str)

    def __init__(self, folder):
        QThread.__init__(self)
        self.folder = folder

    def run(self):
        while True:
            # 이미지 파일이 하나 더 생성되면 실행
            for filename in os.listdir(self.folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(self.folder, filename)
                    # 이미지 파일 처리 (추가 작업 가능)
                    print("새로운 이미지 파일:", image_path)
                    self.imageCreated.emit(image_path)
                    time.sleep(1)  # 예시로 1초 동안 대기

            time.sleep(1)  # 1초마다 폴더를 확인하기 위한 대기 시간

# GUI를 관리하는 클래스
class MainWindow(QObject):
    def __init__(self):
        QObject.__init__(self)

        self.folder_path = "이미지_폴더_경로"
        self.thread1 = ImageProcessingThread(self.folder_path)
        self.thread2 = ImageCreationThread(self.folder_path)

        # 스레드의 시그널 연결
        self.thread1.imageProcessed.connect(self.onImageProcessed)
        self.thread2.imageCreated.connect(self.onImageCreated)

        # 스레드 시작
        self.thread1.start()
        self.thread2.start()

    @pyqtSlot(str)
    def onImageProcessed(self, image_path):
        # 딥러닝 모델 실행 후 처리 작업
        print("처리 완료된 이미지:", image_path)

    @pyqtSlot(str)
    def onImageCreated(self, image_path):
        # 이미지 파일 생성 후 처리 작업
        print("새로운 이미지 파일 생성됨:", image_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())