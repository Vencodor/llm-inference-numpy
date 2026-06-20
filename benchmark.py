import time
import numpy as np
import inference as kvcache
import infernce_kvcache as no_kvcache

def benchmark(gen_fn, prompt, n_tokens, runs=3):
    times = []
    for _ in range(runs):
        # 캐시 버전이면 매번 초기화 (이전 캐시 안 섞이게)
        
        t = time.perf_counter()
        gen_fn(prompt, token=n_tokens)
        times.append(time.perf_counter() - t)
    return min(times)   # 최솟값 (노이즈 줄이려고)

# 길이별로 두 버전 비교
print(f"{'토큰':>6} {'캐시O':>10} {'캐시X':>10} {'배속':>8}")
for n in [10, 30, 60, 100, 150]:
    t_cache = benchmark(kvcache.generate, "hello", n)
    t_nocache = benchmark(no_kvcache.generate, "hello", n)
    speedup = t_nocache / t_cache
    print(f"{n:>6} {t_cache:>9.2f}s {t_nocache:>9.2f}s {speedup:>7.1f}x")