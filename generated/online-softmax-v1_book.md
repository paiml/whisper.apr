# online-softmax-v1

**Version:** 1.0.0

Online softmax — single-pass max+sum via running normalizer (Milakov & Gimelshein 2018)

## References

- Milakov & Gimelshein (2018) Online normalizer calculation for softmax
- Rabe & Staats (2022) Self-attention Does Not Need O(n²) Memory

## Dependencies

- [softmax-kernel-v1.yaml](softmax-kernel-v1.yaml.md)

## Dependency Graph

```mermaid
graph LR
    online_softmax_v1["online-softmax-v1"] --> softmax_kernel_v1.yaml["softmax-kernel-v1.yaml"]
```

## Equations

### online_normalizer

$$
Online update rule (streaming max + sum_exp):
  Given running state (m_{i-1}, d_{i-1}) and new score x_i:
    m_i = max(m_{i-1}, x_i)
    d_i = d_{i-1} · \exp(m_{i-1} - m_i) + \exp(x_i - m_i)
Final: softmax(x)_j = \exp(x_j - m_n) / d_n

$$

**Domain:** $x \in \mathbb{R}^n, n \geq 1$

**Codomain:** $\sigma(x) \in (0,1)^n, \sum \sigma(x)_i = 1$

**Invariants:**

- $d_i > 0 for all i (sum of positive exponentials)$
- $m_i = max(x_1, ..., x_i)$
- $d_i = \sum_{j=1}^{i} \exp(x_j - m_i)$

### standard_softmax

$$
\sigma(x)_i = \exp(x_i - max(x)) / \sum_j \exp(x_j - max(x))
$$

**Domain:** $x \in \mathbb{R}^n$

**Codomain:** $(0,1)^n$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | equivalence | Online matches standard softmax | $\|online_softmax(x) - standard_softmax(x)\| < \varepsilon element-wise$ |
| 2 | invariant | Output sums to 1 | $\|\sum \sigma(x)_i - 1.0\| < \varepsilon$ |
| 3 | invariant | All outputs strictly positive | $\sigma(x)_i > 0 for all i$ |
| 4 | monotonicity | Order preservation | $x_i > x_j ⟹ \sigma(x)_i > \sigma(x)_j$ |
| 5 | invariant | Shift invariance | $softmax(x + c) = softmax(x) for any scalar c$ |
| 6 | invariant | Two-pass (not three) | $Reads scores array exactly twice: once for online max+sum, once for normalize$ |

## Kernel Phases

1. **online_scan**: Single pass computing running (max, sum_exp) pair — *After processing x_1..x_i: m = max(x_1..x_i), d = Σ exp(x_j - m)*
2. **normalize**: Single pass computing weights[j] = exp(scores[j] - m) / d — *Each weight computed from final (m, d) state*

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-OSM-001 | Equivalence to standard softmax | \|online_softmax(x) - standard_softmax(x)\| < 1e-5 element-wise | Online normalizer update has numerical drift |
| FALSIFY-OSM-002 | Sum-to-one | \|Σ online_softmax(x)_i - 1.0\| < 1e-6 | Normalizer denominator computation error |
| FALSIFY-OSM-003 | Positivity | online_softmax(x)_i > 0 for all i | Underflow in exp() not handled |
| FALSIFY-OSM-004 | Shift invariance | \|online_softmax(x + c) - online_softmax(x)\| < 1e-6 | Max subtraction not properly applied |
| FALSIFY-OSM-005 | Decoder attention dimensions | Correct for kv_len in {1, 6, 64, 448, 1500} | Edge case at specific sequence lengths |
| FALSIFY-OSM-006 | Single-element softmax | online_softmax([x]) = [1.0] for any finite x | Base case handling |

