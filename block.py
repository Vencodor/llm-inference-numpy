import numpy as np
import tiktoken

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

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

def attention(x: np.ndarray, attn_w, attn_b, proj_w, proj_b):
    qkv = x @ attn_w + attn_b # 목표 shape: (6, 2304)

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
        x = x.reshape(seq_len, n_head, d_head)
        # ② transpose: (seq, 12, 64) → (12, seq, 64)
        x = x.transpose(1, 0, 2)
        return x

    q = split_heads(q) #(12, 6, 64)
    k = split_heads(k)
    v = split_heads(v)

    scores = q @ k.transpose(0, 2, 1)
    scores = scores / np.sqrt(64)

    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    scores = scores + mask * (-1e9)

    attn = softmax(scores)

    attn_out = attn @ v

    attn_out = attn_out.transpose(1, 0, 2)   # (12,6,64) → (6,12,64)
    attn_out = attn_out.reshape(seq_len, d_model)  # (6,12,64) → (6,768)

    out = attn_out @ proj_w + proj_b
    return out

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

def block(x, w, i):
    # w: gpt2_all.npz, i: 블록 번호
    # ln_1 → attention → +x
    a = attention(
        layernorm(x, w[f"b{i}.ln1_g"], w[f"b{i}.ln1_b"]),
        w[f"b{i}.attn_w"], w[f"b{i}.attn_b"],
        w[f"b{i}.attn_proj_w"], w[f"b{i}.attn_proj_b"],
    )
    x = x + a
    # ln_2 → mlp → +x
    m = mlp(
        layernorm(x, w[f"b{i}.ln2_g"], w[f"b{i}.ln2_b"]),  # ← ln2! 아까 버그
        w[f"b{i}.fc_w"], w[f"b{i}.fc_b"],
        w[f"b{i}.mlp_proj_w"], w[f"b{i}.mlp_proj_b"],
    )
    x = x + m
    return x

def tokenizer(text):
    return enc.encode(text)

def embedding(ids, w):
    wte = w["wte"]   # (50257, 768)
    wpe = w["wpe"]   # (1024, 768)

    token_emb = wte[ids] #토큰 임베딩
    pos_emb = wpe[np.arange(len(ids))] #위치 임베딩

    return token_emb + pos_emb

W = np.load("gpt2_all.npz")
enc = tiktoken.get_encoding('gpt2')

def gpt2_forward(ids) -> np.ndarray:
    x = embedding(ids, W)

    for i in range(12):
        x = block(x, W, i)

    x= layernorm(x, W['lnf_g'], W['lnf_b'])

    logits = x @ W["wte"].T

    return logits

def generate(prompt, n_tokens=20, temperature=1.0, top_k=5):
    ids = tokenizer(prompt)
    
    for _ in range(n_tokens):
        logits = gpt2_forward(ids)
        next_logits = logits[-1]

        probs = softmax(next_logits / temperature)
        top_ids = np.argsort(probs)[-top_k:]
        top_probs = probs[top_ids]
        top_probs = top_probs / sum(top_probs)

        next_id = np.random.choice(top_ids, p=top_probs)
        
        ids.append(int(next_id))
    
    return enc.decode(ids)

print(generate("The sky is a", n_tokens=100))