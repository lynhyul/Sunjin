import os
# import cv2
import shutil
import glob
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.action_chains import ActionChains

def create_folder(directory):
    # 폴더 생성 함수
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

class NAS_Api:
    def __init__(self, address: str):
        self.options = webdriver.ChromeOptions()
        # skip ssl
        self.options.add_experimental_option('excludeSwitches',["enable-logging"])
        self.options.add_argument("--ignore-certificate-errors")
        self.c_path = "C:/Users/SEHWANG/anaconda3/chromedriver.exe"
        self.driver = webdriver.Chrome(self.c_path,chrome_options=self.options)
        self.count = 0
        # self.driver.implicitly_wait(3)
        self.address = address

    def teardown(self):
        self.driver.quit()

    def auto_test(self,user_id,user_pw):
        # self.driver.get(self.address)
        if self.count == 0 :
            self.driver.get(self.address)
            self.driver.set_window_size(974, 1040)
        self.count += 1
        # while(True):
        #     try :
        self.driver.implicitly_wait(3)
        self.driver.find_element(By.CLASS_NAME,'x-form-text.x-form-field.textfield').send_keys(user_id)
        self.driver.find_element(By.ID,'ext-comp-1026').find_element(By.CLASS_NAME,'x-form-text.x-form-field.textfield').send_keys(user_pw)
        self.driver.find_element(By.ID,'login-btn').click()
        self.driver.find_element(By.CLASS_NAME,'launch-icon.classical.transition-cls').click()
        self.driver.find_element(By.CLASS_NAME,'webfm-file-type-icon').click()
        self.driver.find_element(By.CLASS_NAME,'x-tree-ec-icon.x-tree-elbow-plus').click()
        time.sleep(2)
        self.driver.find_element(By.CLASS_NAME,'x-tree-node-el.x-unselectable.x-tree-node-collapsed').find_element(By.ID,'extdd-37').click()
            

def NAS_Upload(folder_path,user_id,user_pw) :
    flist = os.listdir(folder_path)
    # url = 'https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com/'
    url = 'https://115.94.24.54:5003/'
    test_address = url
    test = NAS_Api(test_address)
    while True :
        # try :
            test.auto_test(user_id,user_pw)
        # except :
        #     pass
    # for file in flist :
    #     file_path = folder_path + '/' + file
        



        
NAS_Upload('c:/xray/0 (2)/','sehwang','tjddms123!')