KV cache
짧은 시퀀스에서 오히려 손해. 매 decode 스텝 concat·재할당 비용은 고정으로 붙는 반면, 캐시가 줄이는 attn 절약분은 시퀀스가 길어야 커짐
(attn은 전체의 11%, 짧을 땐 quadratic 항≈0). → 이득과 비용이 교차하는 crossover 지점이 존재.

Logits
지연의 70%는 logits projection. 연산량(FLOPs)은 MLP가 더 크지만, 
logits 가중치(≈154MB)를 토큰 하나당 통째로 읽어야 해서 arithmetic intensity가 바닥 → memory-bound.

결론
추론은 memory-bound. 구조 최적화만으론 한계, 하드웨어 co-design이 중요.
→ 8주차 quantization은 가장 무거운 weight(logits·mlp)를 int8화해 메모리 트래픽을 직접 침.