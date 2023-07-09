import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


fontpath = "fonts/gulim.ttc"
# image_file = IMP.Tool.Image_Load("D:/test_image/img2.jpg",1)
fontprop  = fm.FontProperties(fname=fontpath, size=10)

# 데이터 예시
categories = ['바른농장', '돌돌말이 대패구이', '선진포크 한돈(온라인)', '한우 1등급 와이즐리',
              '선진포크 한돈(이마트)','푸드머스 대용량','푸드머스','현대 별미육찬','선진포크 한돈(현대)',
              '홈플러스 돌돌말이 앞다리','홈플러스 돌돌말이 삼겹살']
values = [100, 590, 593, 383, 227,738,13288,11,1027,2098,5414]

plt.title('막대 그래프 예시', fontproperties=fontprop)
plt.xlabel('카테고리', fontproperties=fontprop)
plt.ylabel('값', fontproperties=fontprop)
plt.show()