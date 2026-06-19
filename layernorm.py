import numpy as np

def layernorm(x: np.ndarray, gamma, beta, eps=1e-5):
    # 1단계: 각 토큰(행)의 평균과 분산
    #   힌트: x.mean(axis=?, keepdims=True)
    #         axis는 "768개 숫자 방향"으로 평균내야 함. 행 방향이 axis=1
    #         keepdims=True 는 모양을 (seq,1)로 유지해서 다음 뺄셈이 맞게 함
    mean = x.mean(axis=1, keepdims=True)
    var  = x.var(axis=1, keepdims=True)
    
    # 2단계: 정규화 (평균 빼고 표준편차로 나누기)
    #   힌트: 표준편차 = sqrt(var + eps).  eps는 0으로 나누기 방지용
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # 3단계: gamma 곱하고 beta 더하기
    out = x_norm * gamma + beta
    
    return out

# 테스트
w = np.load("ln_weights.npz")
gamma, beta = w["gamma"], w["beta"]

# 1주차에서 만든 x (6, 768) 를 여기서 다시 만들거나 불러와서 테스트
x = np.random.randn(6, 768).astype(np.float32)  # 일단 더미로 테스트
out = layernorm(x, gamma, beta)

print("출력 shape:", out.shape)              # (6, 768) 유지돼야 함
print("정규화 후 첫 토큰 평균:", out[0].mean())  # gamma,beta 때문에 정확히 0은 아님