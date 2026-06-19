from transformers import GPT2LMHeadModel
import numpy as np

model = GPT2LMHeadModel.from_pretrained("gpt2")
sd = model.state_dict()

weights = {}

# 임베딩
weights["wte"] = sd["transformer.wte.weight"].numpy()
weights["wpe"] = sd["transformer.wpe.weight"].numpy()

# 12개 블록 각각
for i in range(12):
    p = f"transformer.h.{i}."
    # ln_1 (attention 앞)
    weights[f"b{i}.ln1_g"] = sd[p+"ln_1.weight"].numpy()
    weights[f"b{i}.ln1_b"] = sd[p+"ln_1.bias"].numpy()
    # attention
    weights[f"b{i}.attn_w"] = sd[p+"attn.c_attn.weight"].numpy()
    weights[f"b{i}.attn_b"] = sd[p+"attn.c_attn.bias"].numpy()
    weights[f"b{i}.attn_proj_w"] = sd[p+"attn.c_proj.weight"].numpy()
    weights[f"b{i}.attn_proj_b"] = sd[p+"attn.c_proj.bias"].numpy()
    # ln_2 (MLP 앞)  ← 아까 빠졌던 것
    weights[f"b{i}.ln2_g"] = sd[p+"ln_2.weight"].numpy()
    weights[f"b{i}.ln2_b"] = sd[p+"ln_2.bias"].numpy()
    # mlp
    weights[f"b{i}.fc_w"] = sd[p+"mlp.c_fc.weight"].numpy()
    weights[f"b{i}.fc_b"] = sd[p+"mlp.c_fc.bias"].numpy()
    weights[f"b{i}.mlp_proj_w"] = sd[p+"mlp.c_proj.weight"].numpy()
    weights[f"b{i}.mlp_proj_b"] = sd[p+"mlp.c_proj.bias"].numpy()

# 마지막 최종 LayerNorm (12블록 다 통과한 뒤)
weights["lnf_g"] = sd["transformer.ln_f.weight"].numpy()
weights["lnf_b"] = sd["transformer.ln_f.bias"].numpy()

np.savez("gpt2_all.npz", **weights)
print("저장 완료. 총 key 개수:", len(weights))