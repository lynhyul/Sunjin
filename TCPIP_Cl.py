import socket
import time

# 서버 호스트와 포트
HOST = socket.gethostbyname(socket.gethostname())
print(HOST)
PORT = 3333

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect((HOST, PORT))
while True :
    msg = input('msg:') # 서버로 보낼 msg 입력
    socket.sendall(msg.encode(encoding='utf-8'))
    # # 서버가 에코로 되돌려 보낸 메시지를 클라이언트가 받음
    # data = socket.recv(100)
    # msg = data.decode() # 읽은 데이터 디코딩
    # print('echo msg:', msg)

socket.close()