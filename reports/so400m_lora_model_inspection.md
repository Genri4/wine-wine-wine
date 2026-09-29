# SigLIP2 SO400M LoRA model inspection

**Inspection date:** 2026-09-24
**Checkpoint:** `google/siglip2-so400m-patch14-384`
**Pinned revision:** `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`
**Loader:** `transformers.AutoModel` / `AutoImageProcessor`
**Installed Transformers:** `5.17.0`
**Inspection environment:** local cached checkpoint, offline mode, RTX 4060 8 GB

## Observed implementation

The checkpoint loads as `transformers.models.siglip.modeling_siglip.SiglipModel`; the image processor loads as `transformers.models.siglip.image_processing_siglip.SiglipImageProcessor`. The processor uses a 384×384 resize, RGB mean/std `(0.5, 0.5, 0.5)`, and no center crop, matching the frozen recognition pipeline.

| Property | Observed value |
|---|---:|
| Total checkpoint parameters | 1,136,008,498 |
| Vision model parameters | 428,225,600 |
| Vision transformer blocks | 27 (indices 0–26) |
| Vision hidden dimension | 1,152 |
| Vision MLP intermediate dimension | 4,304 |
| Attention heads | 16 |
| Image / patch size | 384 / 14 |
| Runtime model weight dtype | float16 |
| Global image embedding | pooled vision output, 1,152 dimensions, L2-normalized |

The exact last-block module names were enumerated from the loaded model. Blocks 23–26 each contain these `nn.Linear` attention projections, all with shape 1152×1152:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.out_proj
```

Each of those four blocks also has `mlp.fc1` (1152→4304) and `mlp.fc2` (4304→1152); those MLP modules are not LoRA targets. The SigLIP text model is a separate child of `SiglipModel` and is excluded from the image-only objective and LoRA injection.

## Selected implementation scope

The controlled run uses one configuration: rank 8, alpha 16, dropout 0.05; LoRA is attached only to the 16 observed attention projections in vision blocks 23–26. This adds **294,912 trainable parameters**. Every checkpoint parameter outside those LoRA matrices remains frozen. No fused QKV projection or inferred module name is used.

## Inspection caveat

Context7 was unavailable after two connection attempts, so this inspection uses the actual installed Transformers package and the locally cached, revision-pinned model rather than remote documentation. Transformers emitted warnings about the checkpoint's text BOS/EOS token ids when loading; the text tower is unused by image feature extraction and is not adapted.
