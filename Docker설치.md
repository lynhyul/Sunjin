## 1. Docker 설치

1. **Docker**는 가상 머신처럼 독립된 실행환경을 만들어주는 것으로, 운영체제를 설치하 것과 유사한 효과를 낼 수 있지만, 실제 운영체제를 설치하지 않기 때문에 설치 용량이 적고 실행 속도 또한 빠릅니다. 예전에는 윈도에 VM Ware와 같은 가상 머신을 설치하였으나 최근에는 리눅스 계열에서 Docker가 그 역할을 대신하고 있습니다. 

   

2. 도커의 사용 이유는 다음과 같습니다.

   - **구성 단순화** Docker는 하나의 Configuration으로 모든 플랫폼에서 실행할 수 있습니다. Configuration 파일을 코드에 넣고 환경 변수를 전달하여 다른 환경에 맞출 수 있습니다. 따라서 하나의 Docker 이미지를 다른 환경에서 사용할 수 있습니다.

   - **코드 관리** Docker는 일관된 환경을 제공하여 개발 및 코딩을 훨씬 편안하게 만들어줍니다. Docker 이미지는 변경이 불가하기에 개발환경에서 운영 환경까지 애플리케이션 환경이 변경되지 않는 이점이 존재합니다.
   - **개발 생산성 향상** 개발 환경을 운영 환경에 최대한 가깝게 복제할 수 있습니다. Docker를 사용하면 코드가 운영 환경의 컨테이너에서 실행될 수 있으며 VM과 달리 Docker는 오버 헤드 메모리 용량이 적기에 여러 서비스를 실행하는데 도움이 됩니다. 또한 Docker의 Shared Volume을 사용하여 호스트에서 컨테이너의 어플리케이션 코드를 사용할 수 있도록 할 수 있습니다. 이를 통해 개발자는 자신의 플랫폼 및 편집기에서 소스 코드를 편집할 수 있으며 이는 Docker내에서 실행 중인 환경에 반영됩니다.
   - **애플리케이션 격리** Web Server(e.g. Apache, Nginx)와 연결된 API 서버를 격리할 필요가 있는 경우가 있습니다. 이 경우 다른 컨테이너에서 API를 서버를 실행할 수 있습니다.
   - **빠른 배포** 컨테이너가 OS를 부팅하지 않고 어플리케이션을 실행하기 때문에 Docker 컨테이너를 매우 빠르게 만들 수 있습니다.



3.  Docker는 Client (docker)와 서버 (dockerd)로 구성되어 있습니다. Docker Images는 read only의 docker container를 생성하기 위한 template이고, Container는 images가 실제 메모리에 로딩된 instance입니다. 하나의 images로 유사한 container를 만들 수 있습니다. Registry는 Docker hub이며 images의 저장소입니다. 
    - Images: libs와 package의 template, read only
    - Container: Images가 설치되어 메모리에 로딩된 instance ![img](https://blog.kakaocdn.net/dn/bviqtp/btqLkg4LRhJ/qo74pSME2KtAMavs9LXS20/img.png)





GPG키 및 저장소 추가를 진행 해줘야 합니다.

* GPG(GnuPG) : **비대칭 개별 키 쌍을 사용하여 메시지를 암호화**합니다. Campaign을(를) 사용하여 GPG 암호화를 구현하려면 Campaign 컨트롤 패널에서 관리자가 직접 마케팅 인스턴스에 GPG 키를 설치 및/또는 생성해야 합니다.



#### 1) 에러발생

~~~
docer ps
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
~~~

이유는 WSL에서 실행하려면 다른 방법으로 접근해야 한다는 것이었습니다.

https://docs.docker.com/desktop/windows/wsl/#download 에 접속해서 [Windows용 Docker 데스크톱을](https://desktop.docker.com/win/main/amd64/Docker Desktop Installer.exe) 다운로드를 해줘야 합니다.



1. 설치되면 Windows 시작 메뉴에서 Docker Desktop을 시작한 다음 작업 표시줄의 숨겨진 아이콘 메뉴에서 Docker 아이콘을 선택합니다. 아이콘을 마우스 오른쪽 단추로 클릭하여 Docker 명령 메뉴를 표시하고 "설정"을 선택합니다. ![Docker 데스크톱 대시보드 아이콘](https://learn.microsoft.com/ko-kr/windows/wsl/media/docker-starting.png)
2. **설정**>**일반**에서 "WSL 2 기반 엔진 사용"이 선택되어 있는지 확인합니다. ![Docker Desktop 일반 설정](https://learn.microsoft.com/ko-kr/windows/wsl/media/docker-running.png)
3. Docker 통합을 사용하도록 설정하려는 설치된 WSL 2 배포판에서 선택합니다. **설정**>**리소스**>**WSL 통합**. ![Docker Desktop 리소스 설정](https://learn.microsoft.com/ko-kr/windows/wsl/media/docker-dashboard.png)
4. Docker가 설치되었는지 확인하려면 WSL 배포판(예: Ubuntu)을 열고 다음을 입력하여 버전 및 빌드 번호를 표시합니다. `docker --version`
5. 다음을 사용하여 간단한 기본 제공 Docker 이미지를 실행하여 설치가 올바르게 작동하는지 테스트합니다. `docker run hello-world`



설치가 완료되면 다음과 같이 정상적으로 실행 되는 것을 확인 할 수 있었습니다.

~~~
lynhyul@DESKTOP-TEUDVF9:~$ docker version
Client: Docker Engine - Community
 Version:           20.10.21
 API version:       1.41
 Go version:        go1.18.7
 Git commit:        baeda1f
 Built:             Tue Oct 25 18:01:58 2022
 OS/Arch:           linux/amd64
 Context:           default
 Experimental:      true

Server: Docker Desktop
 Engine:
  Version:          20.10.21
  API version:      1.41 (minimum version 1.12)
  Go version:       go1.18.7
  Git commit:       3056208
  Built:            Tue Oct 25 18:00:19 2022
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          1.6.10
  GitCommit:        770bd0108c32f3fb5c73ae1264f7e503fe7b2661
 runc:
  Version:          1.1.4
  GitCommit:        v1.1.4-0-g5fd4c4d
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
lynhyul@DESKTOP-TEUDVF9:~$ docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
~~~



### 2.Container 생성 (`docker-compose.yml` 작성)

빌드된 이미지를 이용해서 `container`를 생성합니다. `cli`를 통해 `container`에 필요한 arguments를 직접 입력해주는 `docker run`과, 미리 `docker-compose.yml` 파일에 arguments를 모두 입력해놓고 실행 시 불러오는 `docker compose`가 있습니다.

#### - docker run

docker에서 가장 많이 사용하게 되는 `run`에는 다양한 option들이 있습니다. 그 중, 몇가지만 간략하게 확인하자.

- `-v`, `--volume` : [host-src]:[container-dest] 저장 공간 bind
- `-d`, `--detach` : 백그라운드 실행
- `-p`, `--port` : [host-port]:[container-port] 포트 포워딩
- `--gpus` : 사용할 gpu 입력 (ex1. ‘“device=0,2”’) (ex2. all)
- `--rm` : container 상태가 exit이 되면 자동으로 삭제



#### 3. Pytorch 실행 해보기

~~~
docker pull pytorch/pytorch			## Docker 이미지 생성
sudo docker run -it pytorch/pytorch		## Docker 컨테이너 생성
~~~

위와 같이 이미지를 생성 한 후에, 컨테이너를 생성하면 다음과 같이 workspace가 활성화 됩니다.

~~~
root@38e87d991f94:/workspace#
~~~

이 상태에서 Vscode와 연동 하는 방법은 아래와 같습니다.

(참조 : https://shuka.tistory.com/18)

1) Extensions에서 Remote Development 설치

VSCode에서 docker container을 연동해서 사용하기 위해 우선 왼쪽 메뉴의 Extension에서 Remote Development를 설치한다.

![img](https://blog.kakaocdn.net/dn/6tcNE/btrgnroFfkk/8HF20FcIvTw6wtvkuT2Eo1/img.png)



2)Remote-Containers: Attach to Running Container...

Ctrl + Shift + p를 누르면 여러 명령어들을 볼 수 있는데 remote-Containers:Attach to Running Container를 선택해 준다.

![img](https://blog.kakaocdn.net/dn/bNZwm9/btrgjFVhiwD/5jXd53vkFSWow6DFlcAOYk/img.png)



선택을 했을 때 만약 docker container가 없거나 실행되어 있지 않은 경우 다음과 같은 메세지 창이 뜬다.



![img](https://blog.kakaocdn.net/dn/dGGn77/btrgnrhUNbj/DYhqp02ODAKmrjTGim0tdK/img.png)



 

이럴 경우 docker container를 만들어 주거나 container를 다음과 같이 start를 해주고 다시 하면 된다.

```
docker start <container 이름>
```

 

container가 있다면 다음과 같이 해당 container가 나타나고 여러 container라면 원하는 container를 클릭하면 된다.

![img](https://blog.kakaocdn.net/dn/n67XM/btrgofO6UuC/MB4S5zY5nPocD8Y0YTWTW0/img.png)

원하는 container를 클릭하면 현재 open되어 있는 vscode창 말고 새로 container가 연동되어 있는 vscode 창이 뜬다.



#### 4. YOLO v7 Dokcer로 실행해보기

먼저, 다음과 같은 명령어로 cuda 11.2 와 cudnn8을 설치 해줍니다.

docker pull 명령어를 통해서 가져 올 수 있습니다.

~~~
docker pull nvidia/cuda:11.2.0-cudnn8-devel-centos7
~~~

docker images 명령어를 통해서 다음과 같이 성공적으로 docker image가 생성 된 것을 확인 할 수 있었습니다.

~~~ 
lynhyul@DESKTOP-TEUDVF9:~$ docker images
REPOSITORY        TAG                            IMAGE ID       CREATED         SIZE
pytorch/pytorch   1.10.0-cuda11.3-cudnn8-devel   46961cbf2ac7   13 months ago   14.4GB
hello-world       latest                         feb5d9fea6a5   14 months ago   13.3kB
~~~

그런 다음 받은 이미지를 통해서 실행하고자 다음과 같이 명령어를 넣어서 실행 해봤습니다.

~~~
lynhyul@DESKTOP-TEUDVF9:~$ docker run \
> -d \
> --name torch \
> --gpus all \
>  --ipc=host \
>  -v $(pwd):/workspace \
> pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel \
> python -m http.server
~~~

그러나 아래와 같이 다시 한 번 에러가 발생 했습니다.

~~~ 
bde6112df84f9434b3067f7659b18ef5df2f3282865a91ec5a86e4367edf2722
docker: Error response from daemon: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: initialization error: WSL environment detected but no adapters were found: unknown.
~~~

GPU카드가 달려 있지 않은 상태로 Nvidia와 연동하려 하였기 때문에 생긴 오류로 예측하고 있습니다. 이는 나중에 사내 노트북을 지급 받으면 다시 진행 하겠습니다.



#### 5. imagefile 넣어보기

---

![image-20221208153636537](C:\Users\lynhy\AppData\Roaming\Typora\typora-user-images\image-20221208153636537.png)

WSL2의 경우, 폴더에 다음과 같이 접속 할 수 있다.

~~~
\\wsl$ 
~~~

위와 같은 방법으로 접근 후에, Ubuntu로 들어 간 다음에 home/{UserName}/ 에 폴더 하나를 만들고 파일을 넣어 둔다.

그런다음에 다음과 같은 명령어로 원하는 컨테이너의 폴더에 추가 입력을 해주면 이미지 파일을 컨테이너에 넣을 수 있다.

~~~
lynhyul@DESKTOP-24K6OHF:~$ sudo docker ps
[sudo] password for lynhyul:
CONTAINER ID   IMAGE           COMMAND   CREATED          STATUS         PORTS     NAMES
bd0050a9078f   detectron2:v0   "bash"    34 minutes ago   Up 6 minutes             detectron2

lynhyul@DESKTOP-24K6OHF:~$ sudo docker cp /home/lynhyul/imagefile/. detectron2:/home/appuser/detectron2_repo/imagefiles/
~~~

다음으로 demo파일을 실행 한 뒤 결과가 출력이 되는지 확인 하고자 하였다.



~~~
appuser@bd0050a9078f:~/detectron2_repo/demo$ python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input ../imagefiles/3.jpg --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50
_FPN_3x/137849600/model_final_f10217.pkl
~~~



그러나 다음과 같은 에러가 발생했다.

~~~
(COCO detections:333): Gtk-WARNING **: 07:35:31.943: cannot open display: :1
/usr/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
~~~



아무래도, WSL Docker 환경에서 별도의 View 프로그램이 없으면 이미지 파일을 띄울 수 없어서 그런 것 같다.

그래서 아래와 같은 방법으로 해결하였다.



1. x server 설치
   아래 사이트에서 x server를 다운받고 설치한다.
   https://sourceforge.net/projects/vcxsrv

wsl2 기준으로 1,3번째 옵션을 체크하고 2번째 옵션은 해제 한다. Additional parameters 부분에 -ac를 추가한다. 안해도 상관은 없다.

설정이 완료되고 마치면 x server가 실행이 되는데, 그 전에 설정 파일을 저장하는 것이 좋다.

컴퓨터를 키면 x server가 실행되도록 하려면 설정파일을 시작 프로그램 폴더에 두면 된다. 시작 프로그램 폴더는 파일 탐색기 주소창에 shell:startup 경로를 입력해서 이동할 수 있다.

2. 디스플레이 환경변수 설정
   우분투에 windows x server를 설치할 때 설정했던 display를 설정하기 위해 아래의 소스를 ~/.bashrc 혹은 ~/.profile 파일에 아래 설정을 붙여넣는다.

export DISPLAY="`grep nameserver /etc/resolv.conf | sed 's/nameserver //'`:0"
export LIBGL_ALWAYS_INDIRECT=1

3. 공용 네트워크 방화벽 해제
   2 까지 하고 하고 intellij를 띄워려 했을때 에러가 나 꽤 삽질을 했다. 결론적으로 공용 네트워크를 꺼야한다. 방화벽을 끄지 안으면 네트워크로 연결된 x server를 실행하지 못해 GUI 프로그램을 띄우지 못한다.



~~~
## Ubuntu

lynhyul@DESKTOP-24K6OHF:~$ cat /etc/resolv.conf
# This file was automatically generated by WSL. To stop automatic generation of this file, add the following entry to /etc/wsl.conf:
# [network]
# generateResolvConf = false
nameserver 172.30.192.1
lynhyul@DESKTOP-24K6OHF:~$
~~~



~~~
## Docker container (running)

appuser@bd0050a9078f:~/detectron2_repo/demo$ export DISPLAY=172.30.192.1:0
~~~

![image-20221208164315983](C:\Users\lynhy\AppData\Roaming\Typora\typora-user-images\image-20221208164315983.png)



우선, detectron2는 여기까지 하는것으로 마무리 하고, 다음에는 YOLOv7을 라벨링까지 해서 학습 하는 단계까지 해보기로 한다.



