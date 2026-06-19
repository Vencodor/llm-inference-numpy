import numpy as np

w = np.load("attn_weights.npz")
attn_w, attn_b = w["attn_w"], w["attn_b"]
proj_w, proj_b = w["proj_w"], w["proj_b"]

x = np.random.randn(6, 768).astype(np.float32)  # 더미 입력 (토큰 6개)

# 1단계: x에 c_attn 곱해서 (seq, 2304) 만들기
#   힌트: 어제 MLP랑 같은 패턴. x @ W + b
qkv = x @ attn_w + attn_b                       # 목표 shape: (6, 2304)

# 2단계: 2304를 768씩 셋으로 쪼개기
#   힌트: np.split(qkv, 3, axis=1) 은 axis=1을 3등분해서 리스트로 줌
#         또는 슬라이싱: qkv[:, :768], qkv[:, 768:1536], qkv[:, 1536:]
q, k, v = np.split(qkv, 3, axis=1)                   # 각각 (6, 768)

n_head = 12
# q, k, v: 각 (6, 768)  (앞 단계에서 만든 것)
seq_len = q.shape[0]      # 6
d_model = q.shape[1]      # 768
d_head = d_model // n_head # 64

def split_heads(x: np.ndarray):
    # x: (seq, 768) → (12, seq, 64)
    # ① reshape: (seq, 768) → (seq, 12, 64)
    #    힌트: x.reshape(seq_len, n_head, d_head)
    x = x.reshape(seq_len, n_head, d_head)
    # ② transpose: (seq, 12, 64) → (12, seq, 64)
    #    힌트: x.transpose(1, 0, 2)  — 축 순서 바꾸기
    x = x.transpose(1, 0, 2)
    return x

q = split_heads(q) #(12, 6, 64)
k = split_heads(k)
v = split_heads(v)

scores = q @ k.transpose(0, 2, 1)
scores = scores / np.sqrt(64)

mask = np.triu(np.ones((seq_len, seq_len)), k=1)
scores = scores + mask * (-1e9)

exp = np.exp(scores - scores.max(axis= -1, keepdims=True))
attn = exp / np.sum(exp, axis= -1, keepdims=True)

attn_out = attn @ v

attn_out = attn_out.transpose(1, 0, 2)   # (12,6,64) → (6,12,64)
attn_out = attn_out.reshape(seq_len, d_model)  # (6,12,64) → (6,768)

out = attn_out @ proj_w + proj_b

print("out shape:", out.shape)   # (12, 6, 64)