# WAPR-DECODE-001: Transcription Pipeline Fix (Poppian Analysis)

**Status**: 🟡 IN PROGRESS
**Last Updated**: December 17, 2025
**Blocking**: Transcription produces repetitive tokens / EOT failure

---

## 1. Methodology: Conjectures and Refutations

> "The method of science is the method of bold conjectures and ingenious and severe attempts to refute them." — Karl Popper, *Objective Knowledge* (1972)

We reject verificationism. We do not seek to prove `whisper.apr` correct. We seek to aggressively falsify its components until the behavior matches the reference implementation.

### 1.1 The Epistemological Framework
1.  **Problem Situation ($P_1$)**: The initial bug or discrepancy.
2.  **Tentative Theory ($TT$)**: A conjecture proposed to explain $P_1$.
3.  **Error Elimination ($EE$)**: Rigorous testing, debugging, and comparison (falsification attempts).
4.  **New Problem ($P_2$)**: The new state arising from the correction of $P_1$, leading to deeper insight.

---

## 2. Previous Cycle ($P_1 \rightarrow TT_1 \rightarrow EE_1$)

**Problem ($P_1$)**: The model attended to padding positions (76-1499) instead of audio content (0-75), resulting in hallucinated output.

**Corroborated Theory ($TT_{H35}$)**: *Positional Singularity*. The absence of an attention mask allowed the decoder to focus on the "empty" silence of padding, which acts as a "super-attractor" due to learned positional embeddings in the original Whisper training (Radford et al., 2023).

**Refutation Attempt ($EE$)**:
-   **Test**: Comparison of attention weights with and without masking.
-   **Result**:
    -   *Without Mask*: 99% attention on padding.
    -   *With Mask*: >95% attention on audio segments.
-   **Status**: $TT_{H35}$ stands. The attention masking fix is correctly implemented.

---

## 3. Current Problem Situation ($P_2$)

Despite the attention mask fix, the system has entered a new failure mode:
1.  **Infinite Repetition**: The model outputs "... sword sword sword" or similar repetitive loops.
2.  **EOT Failure**: The End of Text (EOT) token is never selected, hitting the hard limit `max_tokens` (448).

This resembles "neural text degeneration" described by Holtzman et al. (2019), where maximization-based decoding (greedy search) leads to repetitive loops.

---

## 4. New Conjectures (Hypotheses)

We propose the following bold conjectures to explain $P_2$. We must attempt to falsify them in this order.

### Conjecture A: The Output Projection Divergence ($C_A$)
*Theory*: The final linear layer (`ln_post` + `linear`) projecting hidden states to vocabulary logits has weight mismatches or precision errors compared to `whisper.cpp`.
*Falsification Test*:
1.  Extract the final hidden state $h_{final}$ from `whisper.cpp` for a known input.
2.  Inject $h_{final}$ into `whisper.apr`'s projection layer.
3.  Compare the resulting logits. If $|L_{apr} - L_{cpp}| > \epsilon$, $C_A$ is corroborated.

### Conjecture B: The EOT Suppression Anomaly ($C_B$)
*Theory*: The specific logit for the EOT token (50257) is being artificially suppressed (masked out) or is numerically unstable during the `argmax` step.
*Falsification Test*:
1.  Force the model to predict EOT.
2.  Inspect the raw logit value of token 50257 vs. the top candidate.
3.  If $Logit(EOT)$ is `NaN` or `-inf`, $C_B$ is corroborated.

### Conjecture C: The Greedy Search Pathology ($C_C$)
*Theory*: The implementation is mathematically correct, but the deterministic greedy decoding is falling into a local repetition trap inherent to the model weights when precision differences are present (Vaswani et al., 2017).
*Falsification Test*:
1.  Implement a simple repetition penalty (e.g., set logit of previously generated tokens to $-\infty$).
2.  If generation resolves to coherent text, $C_C$ is corroborated (and the fix is algorithmic).

---

## 5. Peer-Reviewed Citations & References

1.  **Popper, K.** (1963). *Conjectures and Refutations: The Growth of Scientific Knowledge*. Routledge.
    *   *Relevance*: Foundation of our debugging methodology.
2.  **Radford, A., et al.** (2023). "Robust Speech Recognition via Large-Scale Weak Supervision". *OpenAI*.
    *   *Relevance*: Defines the Whisper architecture, positional embeddings, and expected EOT behavior.
3.  **Vaswani, A., et al.** (2017). "Attention Is All You Need". *NeurIPS*.
    *   *Relevance*: Core Transformer architecture, Multi-Head Attention mechanism.
4.  **Holtzman, A., et al.** (2019). "The Curious Case of Neural Text Degeneration". *ICLR*.
    *   *Relevance*: Explains why greedy decoding leads to repetition loops in autoregressive models.

---

## 6. Action Plan (Next Session)

1.  **Falsify $C_A$ (Output Projection)**:
    -   Create `tests/verify_output_projection.rs`.
    -   Load weights, run a dummy vector, compare against stored golden values from `whisper.cpp`.
2.  **Falsify $C_B$ (EOT Probability)**:
    -   Instrument `GreedyDecoder` to print top-5 logits for every step.
    -   Observe the rank of EOT.
3.  **Falsify $C_C$ (Repetition)**:
    -   Temporarily hack `greedy_decoder.rs` to penalize immediate repeats.

---


## Quality Gate Status

```
✅ Linting (clippy)
✅ Unit Tests (test-fast)
✅ Coverage (>90%)
🔴 Integration Tests (ground_truth_tests skipped)
```