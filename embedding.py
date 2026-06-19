import tiktoken
import numpy as np

# 1. weight 불러오기
w = np.load("gpt2_weights.npz")
wte = w["wte"]   # (50257, 768)
wpe = w["wpe"]   # (1024, 768)

# 2. 토큰화
enc = tiktoken.get_encoding("gpt2")
text = "Hello, I am a student"
ids = enc.encode(text)          # 정수 ID 리스트

token_emb = wte[ids]
pos_emb = wpe[np.arange(len(ids))]

x = token_emb + pos_emb

print("token_emb shape:", token_emb.shape)
print("pos_emb shape:", pos_emb.shape)
print("최종 x shape:", x.shape)   # 목표: (토큰수, 768)