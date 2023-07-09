import pandas as pd
import Levenshtein


def calculate_similarity(sentence1, sentence2):
    try :
        idx = sentence2.index(sentence1[:3])
        sentence2 = sentence2[idx:]
    except :
        pass
    # 문장의 유사도 계산
    similarity_ratio = Levenshtein.ratio(sentence1, sentence2)

    # 첫 번째 문장이 두 번째 문장에 포함되는지 확인
    is_contained = sentence1 in sentence2

    return similarity_ratio

def remove_whitespace(text):
    try :
        if '.' in text :
            text = text.replace('.','')
        text = text.replace(' ', '')
    except :
        pass
    return text

######## 표기 사항 오류 탐색 알고리즘#########
df = pd.read_csv('d:/yolo/ref.csv',encoding='cp949')
df = df.applymap(remove_whitespace)
name = list(df['품목명'])
target_count = list(df['계획'])
order = list(df['주문처'])
slaughter = list(df['도축장'])
nums = list(df['묶음번호'])
day = list(df['소비기한'])
farm = list(df['농장명'])
place = list(df['소재지'])
c_num = list(df['인증번호'])

string = '앞다리살찌개500g(냉장)'
for idx,st in enumerate(name) :
    if calculate_similarity(string,st) > 0.85 :
        print(st)

# find_index = name.index(string)



