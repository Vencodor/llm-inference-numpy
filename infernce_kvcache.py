import tiktoken
import numpy as np


def softmax(x, axis=-1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def embedding(w, ids, index=0):
    wte = w['wte']
    wpe = w['wpe']

    token_embed = wte[ids]
    pos_embed = wpe[np.arange(len(ids)) + index]

    return token_embed + pos_embed

def layernorm(x: np.ndarray, g, b):
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)

    x = (x - mean) / np.sqrt(var + 1e-5)

    return x * g + b

def spliter(x: np.ndarray, len1, dims, head=12):
    x = x.reshape(len1, head, dims)   # (len, 12, 64)
    return x.transpose(1, 0, 2)       # (12, len, 64)

def attention(x: np.ndarray, index, attn_w, attn_b, proj_w, proj_b):
    len1 = x.shape[0]
    len2 = x.shape[1]
    dims = len2 // 12

    if kv_cache[index] is None:   # prefill, x=(len,768)
        qkv = x @ attn_w + attn_b
        q, k, v = np.split(qkv, 3, axis=1)

        q = spliter(q, len1, dims)
        k = spliter(k, len1, dims)
        v = spliter(v, len1, dims)

        kv_cache[index] = k, v

        scores = q @ k.transpose(0, 2, 1)
        scores = scores / np.sqrt(64)

        mask = np.triu(np.ones((len1, len1)), k=1) * (-1e9)
        scores = scores + mask
    else:   # decode, x=(1,768)
        k, v = kv_cache[index]

        qkv1 = x @ attn_w + attn_b
        q, k1, v1 = np.split(qkv1, 3, axis=1)

        q = spliter(q, len1, dims)
        k1 = spliter(k1, len1, dims)
        v1 = spliter(v1, len1, dims)

        k = np.concatenate([k, k1], axis=1)
        v = np.concatenate([v, v1], axis=1)

        kv_cache[index] = k, v

        scores = q @ k.transpose(0, 2, 1)
        scores = scores / np.sqrt(64)

    scores = softmax(scores)

    out = scores @ v
    out = out.transpose(1, 0, 2)
    out = out.reshape(len1, len2)

    return out @ proj_w + proj_b

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    ))

def mlp(x, fc_w, fc_b, proj_w, proj_b):
    h = x @ fc_w + fc_b
    h = gelu(h)
    return h @ proj_w + proj_b

def block(w, i, x):
    x = x + attention(
        layernorm(x, w[f"b{i}.ln1_g"], w[f"b{i}.ln1_b"]),
        i,
        w[f"b{i}.attn_w"], w[f"b{i}.attn_b"],
        w[f"b{i}.attn_proj_w"], w[f"b{i}.attn_proj_b"],
    )

    x = x + mlp(
        layernorm(x, w[f"b{i}.ln2_g"], w[f"b{i}.ln2_b"]),
        w[f"b{i}.fc_w"], w[f"b{i}.fc_b"],
        w[f"b{i}.mlp_proj_w"], w[f"b{i}.mlp_proj_b"],
    )

    return x

def gpt2_forward(w, ids, index) -> np.ndarray:
    x = embedding(w, ids, index)

    for i in range(12):
        x = block(w, i, x)

    x = layernorm(x, W['lnf_g'], W['lnf_b'])   # (len, 768)

    return x[-1] @ W["wte"].T   # (768,) @ (768, 50257) = (50257,)

def generate(prompt, token=30, temperature=1.0, top_k=5):
    global kv_cache
    kv_cache = [None for _ in range(12)]
    ids = enc.encode(prompt)

    prefilled = False
    for _ in range(token):
        if not prefilled:
            logit = gpt2_forward(W, ids, 0)
        else:
            logit = gpt2_forward(W, [ids[-1]], len(ids) - 1)

        probs = softmax(logit / temperature)
        top_ids = np.argsort(probs)[-top_k:]
        top_probs = probs[top_ids]
        top_probs = top_probs / sum(top_probs)
        next = np.random.choice(top_ids, p=top_probs)

        ids.append(int(next))
        prefilled = True

    return enc.decode(ids)

enc = tiktoken.get_encoding('gpt2')
W = np.load("gpt2_all.npz")
kv_cache: list = [None for _ in range(12)]
np.random.seed(0)

print(generate("Hello, my name is", token=30))