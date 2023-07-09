from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
import os
import shutil
import time




def create_folder(directory):
    # 폴더 생성 함수
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'c:/test/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    # Call the Drive v3 API
    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))
            
    
    ########################### folder parameter ###########################
    root_path = 'C:/test2/'
    result_path = 'C:/test_result/'
    create_folder(root_path)
    create_folder(result_path)
    flist = os.listdir(root_path)
    ########################################################################
    stop = 0
    while stop == 0 :
        flist = os.listdir(root_path)
        if len(flist) > 0 :
            time.sleep(0.01)
            for fi in flist :
                try : 
                # 특정 폴더 내 파일 업로드
                ## parents에 업로드할 파일의 상위 폴더의 ID를 넣어주면 해당 폴더 안으로 업로드 된다.
                    file_metadata = {'name' : fi, 'parents' : ['1SdvPO2u_UvbdQHRZOePGF-KimKUOyI0V']}
                    shutil.move(f'{root_path}/{fi}', f'{result_path}/{fi}')
                    media = MediaFileUpload(f'{result_path}/{fi}', resumable = True)
                    file = service.files().create(body = file_metadata, media_body=media, fields='id').execute()
                except :
                    pass
if __name__ == '__main__':
    main()