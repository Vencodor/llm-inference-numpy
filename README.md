# llm-inference-numpy

PyTorch 없이 NumPy만으로 GPT-2 small(124M)의 추론 과정을 바닥부터 구현한 프로젝트다. forward pass의 모든 연산(토큰화, 임베딩, LayerNorm, multi-head self-attention, MLP, residual, autoregressive 생성)을 직접 작성하였으며, HuggingFace GPT-2 출력과 대조하여 정확성을 검증하였다. 학습 목표는 추론 최적화(inference optimization)의 동작 원리를 구현 수준에서 내재화하는 데 있다.

## 프로젝트 원칙

- 방향성은 외부에 질의하되, 구현은 직접 수행한다.
- 각 단계에서 동작 원리("왜")를 우선 규명한 뒤 구현한다.
- 결론은 직감이 아니라 측정에 근거한다.
- 백지 재구현을 통해 구조의 내재화를 검증한다.

---

## GPT-2 아키텍처 개요

GPT-2 small(124M)은 디코더-only 트랜스포머다.

| 항목 | 값 |
|---|---|
| 어휘 크기 (vocab) | 50,257 |
| 임베딩 차원 (d_model) | 768 |
| 레이어 수 | 12 |
| 어텐션 헤드 | 12 (헤드당 64차원) |
| 컨텍스트 길이 | 1,024 |
| MLP 은닉 차원 | 3,072 (4 × 768) |

### Forward 흐름

1. **임베딩.** 입력 토큰을 토큰 임베딩(`wte`)으로 변환하고, 위치 임베딩(`wpe`)을 더한다.
2. **디코더 블록 × 12.** 각 블록은 Pre-LN 구조를 따른다.
   - LayerNorm → causal multi-head self-attention → residual add
   - LayerNorm → MLP (Linear 4배 확장 → GELU → Linear 축소) → residual add
3. **최종 LayerNorm.**
4. **출력 projection.** 마지막 토큰 표현에 출력 행렬(`wte`와 가중치 공유)을 곱해 어휘 전체에 대한 logits를 산출한다.
5. **autoregressive 생성.** logits에서 다음 토큰을 샘플링(greedy / temperature / top-k)하고, 생성 토큰을 입력에 추가하여 반복한다.

### 핵심 설계 특징

- **Causal mask.** 각 토큰은 자신 이전의 토큰만 참조한다(미래 토큰 차단).
- **Pre-LN.** 정규화를 서브레이어 입력 단계에 적용하여 학습 안정성을 확보한다.
- **Weight tying.** 입력 임베딩 행렬과 출력 projection 행렬을 공유한다(`wte`).
- **KV cache.** 디코드 단계에서 이전 토큰의 Key·Value를 저장하여 prefix 재계산을 회피한다.

---

## 핵심 교훈

### 1. 추론의 병목은 연산량이 아니라 메모리 대역폭이다

프로파일링 결과 logits projection이 지연의 약 70%를 차지하였다. FLOPs 기준으로는 MLP가 더 크지만, logits 가중치(약 154MB)를 토큰마다 전부 메모리에서 읽어야 하므로 arithmetic intensity가 낮고, 결과적으로 wall-clock을 지배한다. 병목의 본질은 연산이 아니라 메모리 접근이다.

### 2. 최적화는 측정에 선행할 수 없다

KV cache를 구현하였으나 약 30토큰 구간에서는 미적용 버전이 더 빨랐다. KV cache가 절감하는 attention 연산은 전체의 약 11%에 불과하며, concatenation 및 메모리 재할당 등의 고정 비용이 짧은 시퀀스에서 절감분을 상회한다. 따라서 결론은 "KV cache가 우수하다"가 아니라 "특정 시퀀스 길이 이상에서 이득이 발생한다"는 조건부 형태여야 한다.

### 3. 소프트웨어 최적화의 한계와 하드웨어 co-design

memory-bound 특성은 알고리즘만으로 해소되지 않는다. 가중치를 int8로 4배 압축하더라도, int8 연산 유닛이 없는 NumPy 환경에서는 dequantization으로 float를 복원하므로 실측 속도 이득이 없다. 실질적 이득은 int8 GEMM을 네이티브로 지원하는 하드웨어(NPU)에서 발생한다. 추론 최적화가 하드웨어와 분리될 수 없는 이유가 여기에 있다.

### 4. outlier가 양자화 정확도를 좌우한다

per-tensor 방식은 단일 scale을 공유하므로 하나의 outlier가 행렬 전체의 정밀도를 저하시킨다. per-channel 방식은 열마다 scale을 두어 outlier의 영향을 해당 열에 국한시킨다. 출력 logit 하나는 가중치 행렬의 한 열로 계산되므로, 오차를 출력 단위로 격리하려면 scale 단위가 열이어야 한다.

| 방식 | logit 오차 (max diff) |
|---|---|
| per-tensor | 2.7 |
| per-channel | 0.9 ~ 2.29 (입력별 변동) |

### 5. 텐서의 의미는 연산에 의해 결정된다

`wte`는 입력 임베딩(토큰 → 벡터 조회)과 출력 projection(벡터 → 어휘 점수)의 두 역할을 겸한다. 동일한 50,257 차원이 입력에서는 조회 대상 행으로, 출력에서는 점수 산출 열로 기능한다. 행렬의 행/열 의미는 고정된 것이 아니라 적용되는 연산에 따라 정해진다.

### 6. 반복 연산의 벡터화

per-channel scale은 열 단위 반복으로 표현할 수 있으나, `np.max(axis=0, keepdims=True)`를 통해 동일 연산을 단일 축 연산으로 압축할 수 있다. attention의 `q @ k.T` 역시 모든 토큰쌍의 내적을 행렬곱 하나로 압축한 사례다. 논리는 반복으로 설계하되 구현은 축 연산으로 벡터화하는 것이 NumPy 활용의 핵심이다.

### 7. 조기 추상화 지양

양자화 로직을 사전에 클래스로 설계하는 대신 함수로 구현하고, 요구사항이 구체화된 시점에 리팩터링하는 방식을 택했다. per-tensor에서 per-channel로의 확장은 `axis` 인자 하나의 추가로 충분하였다. 확정되지 않은 요구사항에 대한 선제적 추상화는 불필요한 복잡도를 유발한다.

### 8. 직접 구현은 프런티어 연구의 문제의식과 연결된다

본 프로젝트에서 도달한 "KV cache는 memory-bound이며 긴 컨텍스트에서 병목으로 작용한다"는 결론은 KV cache 압축을 다루는 최신 연구(TurboQuant, ICLR 2026)의 문제의식과 일치한다. TurboQuant는 무작위 회전(rotation)으로 outlier를 분산시켜, calibration 없이 KV cache를 약 3비트로 압축한다. 바닥부터의 구현은 최신 연구가 특정 문제를 다루는 이유를 이해하는 기반이 된다.

---

## 8주 로드맵

| 주차 | 내용 |
|---|---|
| 1 | 토큰화·임베딩 (BPE, wte+wpe) |
| 2 | LayerNorm·MLP (GELU) |
| 3 | Attention (QKV, multi-head, causal mask, softmax) |
| 4 | 블록 조립·12층 스택·HF 출력 대조 (최대차 0.00015) |
| 5 | 샘플링·생성 (greedy / temperature / top-k) |
| 6 | KV cache (prefill/decode 분기, 블록별 K·V 캐시) |
| 7 | 측정·프로파일링 (병목 분석, KV cache crossover 확인) |
| 8 | 양자화 (per-tensor / per-channel, 정확도·크기 측정) |

## 양자화 요약

핵심 식:

```
scale = absmax / 127
q     = round(W / scale)        # float32 → int8 (저장 크기 1/4)
W'    = q * scale               # 복원 (dequantize)
```

- 가중치 레벨 오차 상한: `|W − W'| ≤ scale / 2`
- 대상: 대형 가중치 행렬(logits, MLP, attention projection). LayerNorm 등 소형 벡터는 제외(이득이 없고 정밀도에 민감).
- 저장 크기: logits 가중치 기준 약 154MB → 38MB (4배 감소).
- per-tensor와 per-channel의 차이는 변환 대상이 아니라 scale을 두는 단위에 있다. 변환은 전체 원소를 동일하게 int8로 수행한다.