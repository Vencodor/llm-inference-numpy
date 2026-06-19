import tiktoken
import numpy as np


def softmax(x, axis=-1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def embedding(w, ids):
    wte = w['wte']
    wpe = w['wpe']

    token_embed = wte[ids]
    pos_embed = wpe[np.arange(len(ids))]

    return token_embed + pos_embed

def layernorm(x: np.ndarray, g, b):
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)

    x = (x - mean) / np.sqrt(var + 1e-5)

    return x * g + b

def attention(x: np.ndarray, attn_w, attn_b, proj_w, proj_b):
    qkv = x @ attn_w + attn_b #(len, 768*3)
    q,k,v = np.split(qkv, 3, axis=1) #(len, 768)

    head = 12
    len1 = x.shape[0]
    len2 = x.shape[1]
    dims = len2 // head

    def spliter(x: np.ndarray):
        x = x.reshape(len1, head, dims) #(len, 12, 64)
        return x.transpose(1, 0, 2) #(12, len, 64)
    
    q = spliter(q)
    k = spliter(k)
    v = spliter(v)

    scores = q @ k.transpose(0, 2, 1) #(12, len, 64) @ (12, 64, len) = (12, len)
    scores = scores / np.sqrt(64)

    mask = np.triu(np.ones((len1, len1)), k=1)
    scores = scores + mask * (-1e9)

    scores = softmax(scores)

    out = scores @ v #(12, len, 64)
    out = out.transpose(1, 0, 2) #(len, 12, 64)
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
        w[f"b{i}.attn_w"], w[f"b{i}.attn_b"],
        w[f"b{i}.attn_proj_w"], w[f"b{i}.attn_proj_b"],
    )

    x = x + mlp(
        layernorm(x, w[f"b{i}.ln2_g"], w[f"b{i}.ln2_b"]),  # ← ln2! 아까 버그
        w[f"b{i}.fc_w"], w[f"b{i}.fc_b"],
        w[f"b{i}.mlp_proj_w"], w[f"b{i}.mlp_proj_b"],
    )

    return x

def gpt2_forward(w, ids) -> np.ndarray:
    x = embedding(w, ids)

    for i in range(12):
        x = block(w,i, x)
    
    x = layernorm(x, W['lnf_g'], W['lnf_b']) #(len, 768)

    return x @ W["wte"].T #(len, 768) @ (768, 51284?)

def generate(prompt,token = 50, temperature = 1.0, top_k = 5):
    ids = enc.encode(prompt)

    for _ in range(token):
        logit = gpt2_forward(W, ids)[-1]
        
        probs = softmax(logit / temperature)
        top_ids = np.argsort(probs)[-top_k:]
        top_probs = probs[top_ids]
        top_probs = top_probs / sum(top_probs)

        next = np.random.choice(top_ids, p=top_probs)
        ids.append(int(next))

    return enc.decode(ids)

enc = tiktoken.get_encoding('gpt2')
W = np.load("gpt2_all.npz")

print(generate('dooseong is genius.'))