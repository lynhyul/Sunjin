## 1. Docker 설치

1. **Docker**는 가상 머신처럼 독립된 실행환경을 만들어주는 것으로, 운영체제를 설치하 것과 유사한 효과를 낼 수 있지만, 실제 운영체제를 설치하지 않기 때문에 설치 용량이 적고 실행 속도 또한 빠릅니다. 예전에는 윈도에 VM Ware와 같은 가상 머신을 설치하였으나 최근에는 리눅스 계열에서 Docker가 그 역할을 대신하고 있습니다. 

   

2.  도커의 사용 이유는 다음과 같습니다.

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





#### 2. YOLO v7 Dokcer로 실행해보기

먼저, 다음과 같은 명령어로 cuda 11.2 와 cudnn8을 설치 해줍니다.

docker pull 명령어를 통해서 가져 올 수 있습니다.

~~~
docker pull nvidia/cuda:11.2.0-cudnn8-devel-centos7
~~~

