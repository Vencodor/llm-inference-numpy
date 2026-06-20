양자화 (Quantization)
개념. 가중치 행렬의 모든 원소를 float32 → int8로 변환(저장 크기 1/4). 정밀도와 크기를 trade-off. 핵심 식: scale = absmax/127, q = round(W/scale), 복원 W' = q·scale. 오차 상한(가중치 레벨) = scale/2.
대상. 큰 가중치 행렬 — logits, MLP, attention projection. (LayerNorm·작은 벡터는 제외: 이득 없고 정밀도 민감.)
per-tensor vs per-channel. 변환은 전체 원소 동일, 차이는 scale을 두는 단위.

per-tensor: scale 1개 → outlier 하나가 전체 정밀도 파괴.
per-channel(axis=0, 열마다 scale): outlier를 그 열에 격리. 출력 logit 하나 = 가중치 한 열로 계산되므로, 오차를 출력 단위로 가두려면 열 단위 scale이어야 함.
NumPy 구현: np.max(abs(x), axis=0, keepdims=True) — for문(열마다 루프)을 축 연산 한 줄로 벡터화.

실측 (logits 가중치 W['wte'].T, [768, 50257]):
항목값저장 크기154MB → 38MB (4배↓)logit_diff (per-tensor)2.7logit_diff (per-channel)0.9 ~ 2.29 (입력별 변동)
→ per-channel이 일관되게 작음 = outlier 격리 효과를 실측으로 확인.
핵심 결론. NumPy엔 int8 GEMM이 없어 dequant로 float 복원 → 우리 구현은 wall-clock 이득 없음. 양자화의 실제 속도 이득은 int8 연산 유닛을 가진 하드웨어(NPU)에서만 발생. 우리가 얻은 것 = 저장 4배, 오차 정량화, 원리 체득. → 7주차 "memory-bound는 하드웨어 co-design 필요" 결론의 두 번째 증거.

KV cache 양자화. weight 양자화는 정적 가중치 대상. KV 캐시는 (a)online 생성이라 사전 calibration 불가 (b)outlier 심하고 attention 내적에 민감 → 양자화 어려움. KIVI 등 기존 방법 있었고, 터보퀀트(2026)는 회전으로 outlier를 분산시켜 calibration 없이 KV를 ~3비트로 압축한 방법.