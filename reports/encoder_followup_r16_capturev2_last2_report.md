# Encoder follow-up milestone: R16 capacity + Capture-v2 + conditional last2

## Main comparison

| Method | Internal family | Internal full | Hard | Generated | Correct/128 | Gen-hard/64 | Rep/64 | R@5 | Rescue/Break |
|---|---|---|---|---|---|---|---|---|---|
| Frozen |  |  | 1057/1150 | 103/128 | 103 | 40/64 | 63/64 | 128/128 | 0/0 |
| Validated LoRA r8 | 92.40% | 80.93% | 1067/1150 | 107/128 | 107 | 43/64 | 64/64 | 128/128 | 5/1 |
| LoRA r16 | 93.77% | 83.08% | 1066/1150 | 106/128 | 106 | 42/64 | 64/64 | 128/128 | 4/1 |
| Best LoRA + Capture-v2 | 93.77% | 83.08% | 1066/1150 | 106/128 | 106 | 42/64 | 64/64 | 128/128 | 4/1 |
| 50/50 diagnostic |  |  |  | 102/128 | 102 | diagnostic only | diagnostic only | 128/128 | diagnostic only |

## Protocol and interpretation

- Canonical external query batch size: **1** for hard-v2, generated pilot32, and synthetic; catalog reference precompute batch size: **8**.
- Dynamic frozen baseline Top-5 reproduction before adapted ranking: hard **1150/1150**, generated **128/128**, synthetic **4084/4084**.
- Historical Top-5 score floats differ by at most 2.32e-06; accepted numerical tolerance was 3e-6. Top-5 membership and order matched exactly on every query.
- R8 source LoRA-state SHA-256 remains `b3dfce35160cc4c745090f7ec03dd783f86a4c0a82d7566521e7aa21e6a04a04`. R8 and R16 use the exact same train/validation manifests and hard-negative map; only rank/alpha differ (8/16 vs 16/32). Both use Capture-v2 because that is the already frozen R8 domain.
- Selected LoRA + Capture-v2 internal same-family gain over frozen views is **18.24 percentage points**; full-catalog is 83.08% and family margin 0.2013. This is an adapted-vs-frozen gain, not an isolated Capture-v2 gain.
- **Causal limitation:** R8 already used catalog_capture_v2. R16 therefore used the exact same v2 train/validation manifests and hard-negative map; rank/alpha alone changed in the matched R8/R16 pair. R16 is the selected LoRA + Capture-v2 candidate, but no independent v1→v2 effect can be estimated while keeping R8 frozen. The available capacity comparison is matched under v2, not under the originally requested v1 domain.
- RUN D last2 status: not run. Selected LoRA + Capture-v2 is internally conclusive: +18.24pp same-family Top-1, full-catalog 83.08% (frozen 56.22%), margin 0.2013; no deeper unfreeze started.
- Training data access records show catalog references only; benchmark query images, labels, and benchmark-derived hard negatives were not used in training or internal checkpoint selection.

## Generated and benchmark detail

- Frozen generated production: 103/128; validated R8: 107/128; R16: 106/128. Project target: **116/128**; reached: **NO**.
- Best generated-hard: 42/64; representative: 64/64; production R@5: 128/128.
- Best hard-v2: 1066/1150 (92.70%); guard >=1045/1150: **PASS**.
- Best synthetic post-selection: 3984/4084; image-only R@5 4084/4084.
- Generated rescued/broken relative to frozen: 4/1; hard-v2 transitions: 9 net.
- Original frozen 25 generated errors remain immutable (fingerprint `71466010201e9889fe0d7209a850ac366424c6ddd53123ac8e8984ab7b3d0c77`). Target-score wins/losses/ties vs frozen incumbent: R8 11/2/12; R16 6/7/12.
- Best encoder latency: mean 44.00 ms, p95 48.74 ms; full pipeline p95 421.44 ms; frozen encoder p95 48.70 ms; overhead p95 +0.04 ms; SLA <3 sec: True.
- Training cost R16: 6 completed epochs, best epoch 5, summed epoch time 260.9 min, peak 1061 MB. R8 reference run was 6 epochs and 1056 MB peak.
- Recovery: epoch checkpoints under `r16/checkpoints/` include model, optimizer, scheduler, scaler, RNG, validation metrics, and config fingerprint. Resume with `.venv/bin/python scripts/run_so400m_hard_negative_lora.py --rank 16 --alpha 32 --max-epochs 6 --resume artifacts/experiments/encoder_followup_r16_capturev2_last2_20260924T083837Z/r16 --internal-only`.

## Required answers

1. Canonical batch=1 guard added: **YES**. 2. Baseline exact on all 3 sets: **YES** (1150/1150, 128/128, 4084/4084).
3. R8 vs R16 internal winner: **r16**. 4. R16 generated production: **106/128**. 5. Selected LoRA + Capture-v2 internal gain over frozen views: **18.24pp** (not an isolated v2 effect). 6. Incremental Capture-v2 generated gain vs v1: **not separately identifiable**; best R16+v2 is 106/128.
7. Last2 launched: **NO**. 8. Last2 generated result: **N/A**. 9. Best generated: **106/128**. 10. Best generated-hard: **42/64**. 11. Representative: **64/64**.
12. Hard-v2: **1066/1150**. 13. R@5: **128/128 image-only**. 14. Rescued/broken: **4/1**. 15. Original 25 wins/losses/ties: **R8 11/2/12; R16 6/7/12**.
16. Fixed 50/50 fusion diagnostic: generated **102/128**, R@5 **128/128**; no weight search. 17. Main gain attribution: **not causally separable for Capture-v2**; R16 is isolated capacity comparison.
18. Peak VRAM: R16 training **1061 MB**, selected inference **877.32080078125 MB**. 19. R16 training time: **260.9 min**. 20. Test suite: **full suite 335 passed/0 failed (3 warnings); final focused rerun 30 passed/0 failed**. 21. >=116/128 reached: **NO**.

## Final verdict

**G. INVALID_EXPERIMENT** — The benchmark pass and R8/R16 capacity comparison are valid, but the milestone cannot estimate the requested Capture-v1→v2 effect: the frozen R8 control already used catalog_capture_v2, and retraining it was prohibited.

No next encoder milestone was started.
