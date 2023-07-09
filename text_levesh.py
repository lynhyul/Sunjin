import Levenshtein

def calculate_similarity(sentence1, sentence2):
    idx = sentence2.index(sentence1[:3])
    sentence2 = sentence2[idx:]
    # 문장의 유사도 계산
    similarity_ratio = Levenshtein.ratio(sentence1, sentence2)

    # 첫 번째 문장이 두 번째 문장에 포함되는지 확인
    is_contained = sentence1 in sentence2

    return similarity_ratio

# 비교할 문장
sentence1 = '도드람김제에프엠씨'
sentence2 = '도축장:도드람김제애프앰씨'

# 문장의 유사도와 포함 여부 계산
similarity = calculate_similarity(sentence1, sentence2)

# 결과 출력
print(f"Similarity Ratio: {similarity}")

