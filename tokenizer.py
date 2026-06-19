#BPE 방식. 사전에 분석된 단어 표에 맞추어 입력 텍스트들을 쪼개어 ID를 부여한다. ID자체에는 의미가 X

import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = "안녕"
ids = enc.encode(text)

print("토큰 ID:", ids)
print("토큰 개수:", len(ids))
print("각 토큰:", [enc.decode([i]) for i in ids])
