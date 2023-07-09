import pandas as pd 
import os
import numpy as np

### 라벨발행프로그램 쿼리문 ###
inputs = '뒷다리살다짐육1kg'
inputs2 = ['L1230322177355','L1230317177355']
inputs3 = ['도드람']
###############################

print("품명 : ",inputs)
print("묶음 번호 : ",inputs2)
print("도축장 : ",inputs3)

df = pd.read_csv('d:/test.csv',encoding='EUC-KR',index_col=False)
columns = df.columns

if '도축일자' in columns :
    names = df['품명']
    dp = df['도축일자']
    tp = 0
else :
    names = df['품명']
    dp = df['도축장']
    tp = 1

for idx,name in enumerate(names) :
    if name == inputs and tp == 0:
        for num in inputs2 :
            if str(dp[idx]) in num :
                output = num 
    elif name == inputs and tp == 1:
        for num in inputs3 :
            if str(dp[idx]) in num :
                output = num
                            
print("권장 묶음번호 : ", output)





