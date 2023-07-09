import os
import requests
from synology_api import filestation, downloadstation
import shutil
import time

        
class NAS_Api() :
    def __init__(self, username: str,   ## NAS ID
                 password :int,         ## NAS pw
                 nas_ip : str,          ## 원본 이미지 저장 여부
                 nas_port : str,        ## NAS IP ( ex) 192. ..... )
                 src_save : True) :     ## NAS Port ( ex) 5003 )
        ### Nas login ###
        self.username = username        
        self.password = password        
        self.src_save= src_save         
        self.nas_ip = nas_ip            
        self.nas_port = nas_port        
        self.fl = filestation.FileStation(self.nas_ip,       
                                          self.nas_port,     
                                          self.username, 
                                          self.password,
                                          secure=True,
                                          cert_verify=False,
                                          dsm_version=3,
                                          debug=True,
                                          otp_code=None)
    def Upload(self, src_path : str, nas_path : str) :
        if src_path[-1] != '/' :
            src_path = src_path + '/'
        if nas_path[-1] == '/' :
            nas_path = nas_path[:-1]
        idx = list(filter(lambda x: src_path[x] == '/', range(len(src_path))))
        result_path = src_path[:(idx[-2])]+'/result/'
        if self.src_save == True :
            self.create_folder(result_path)
        else : 
            pass
        while True :
            flist = os.listdir(src_path)
            if len(flist) > 0 :
                for file in flist :
                    try :
                        file_path = src_path + file
                        self.fl.upload_file(dest_path=nas_path,file_path=file_path)
                        if self.src_save == True :
                            result_path2 = result_path + file
                            shutil.move(file_path,result_path2)
                        else :
                            os.remove(file_path)
                    except :
                        pass
                time.sleep(3)
                print(f"{len(flist)} file Upload_Complete")
            else :
                pass
        
    def create_folder(self,directory):
        # 폴더 생성 함수
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)


# def main() :
#     nas = NAS_Api(username='sehwang',
#                   password='tjddms123!'
#                   ,nas_ip='115.94.24.54',
#                   nas_port='5003',
#                   src_save=True)
    
#     nas.Upload(src_path='d:/test/',
#                nas_path='/디지털혁신센터/디지털기술팀/01.5_스마트팩토리/03_식육라벨지검사/데이터/')
    
# if __name__ == '__main__':
#     main()