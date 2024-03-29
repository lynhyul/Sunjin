from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os
from selenium.webdriver.common.by import By

# selenium에서 사용할 웹 드라이버 절대 경로 정보
chromedriver = 'c:/chromedriver.exe'
# selenum의 webdriver에 앞서 설치한 chromedirver를 연동한다.
driver = webdriver.Chrome(chromedriver)
# driver로 특정 페이지를 크롤링한다.
driver.get('https://www.google.co.kr/imghp?hl=ko&ogbl') # google화면 띄우기
elem = driver.find_element(By.NAME,"q") # 검색창에 대한 elem
elem.send_keys("가축농장")  # "q"라는 elem(검색창에)에 '비글'이라는 단어를 입력하게 하겠다.
elem.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try :
            driver.find_element(By.CSS_SELECTOR,".mye4qd").click() #더보기 class
        except :
            break
    last_height = new_height 


images = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd") # 이미지에 대한 class name이 있는데
                                                    # class name을 통해서 이미지 지정(커서)
count = 1 
for image in images :
    try :
        image.click()   # 이미지를 클릭하겠다.
        time.sleep(1)
        imgUrl = driver.find_element(By.CSS_SELECTOR,".n3VNCb").get_attribute("src")    
        urllib.request.urlretrieve(imgUrl, "e:/crol/"+ str(count)+ ".jpg")
        count = count + 1
    except :
        pass
driver.close()