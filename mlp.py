import numpy as np

def gelu(x):
    # GELU 근사식 (GPT-2가 쓰는 tanh 버전). 이건 내가 줄게 — 외울 필요 없음
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    ))

def mlp(x, fc_w, fc_b, proj_w, proj_b):
    # x: (seq_len, 768)
    
    # 1단계: Linear1  768 → 3072
    #   힌트: 선형변환은 x @ W + b.  @ 는 행렬곱.
    #         x(seq,768) @ fc_w(768,3072) = (seq,3072).  여기에 fc_b 더하기
    h = x @ fc_w + fc_b
    
    # 2단계: GELU (비선형)
    #   힌트: 위 gelu 함수에 h를 통과시키면 됨
    h = gelu(h)
    
    # 3단계: Linear2  3072 → 768
    #   힌트: 1단계와 같은 패턴. h(seq,3072) @ proj_w(3072,768) + proj_b
    out = h @ proj_w + proj_b
    
    return out

# 테스트
w = np.load("mlp_weights.npz")
x = np.random.randn(6, 768).astype(np.float32)  # 더미
out = mlp(x, w["fc_w"], w["fc_b"], w["proj_w"], w["proj_b"])
print("출력 shape:", out.shape)   # (6, 768) 로 돌아와야 함