
---

## 2026-09-19T21:16:41Z  Modal A100 — consistency check (fair-esm fp32 vs transformers fp16)

```
gpu   : NVIDIA A100-SXM4-40GB
model : facebook/esm2_t33_650M_UR50D
n     : 200 SNCA variants
```

| Metric | Value | Gate |
|---|---|---|
| Spearman ρ | +0.999973 | ≥ 0.999 |
| Max \|diff\| | 0.020073 | — |
| Mean \|diff\| | 0.005289 | — |
| % within 1e-3 | 16.0% | — |

**PASS — implementation reproduces original**
