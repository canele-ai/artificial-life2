# Evaluation Environment

## Modal secrets (available to evaluator.py via modal.Secret.from_name)

Relevant (non-Supabase-preview):
- `anthropic-api-key`       — ANTHROPIC_API_KEY
- `openai-api-key`          — OPENAI_API_KEY
- `canele-github`           — GitHub token (workspace)
- `canele-secrets`          — workspace secret bag
- `canele-eln-secrets`      — workspace secret bag
- `adam-secrets`            — workspace secret bag
- `mochi-secrets`           — workspace secret bag
- `scimarket-secrets`       — workspace secret bag

No `huggingface` secret exists in the workspace — and none is needed (see below).

## Local shell env vars

- ANTHROPIC_API_KEY: missing
- OPENAI_API_KEY: missing
- OPENROUTER_API_KEY: **present**
- GEMINI_API_KEY: missing
- GOOGLE_API_KEY: missing
- HF_TOKEN: missing

## VLM / LLM provider recommendation

**The frozen evaluator requires NO API credentials.** This campaign uses three
locally-loaded vision foundation models (CLIP ViT-B/32, DINOv2-base,
SigLIP-B/16), all downloadable from HuggingFace without authentication:

- `openai/clip-vit-base-patch32`   (public)
- `facebook/dinov2-base`            (public)
- `google/siglip-base-patch16-224`  (public)

Weights are pulled once into a persistent Modal Volume (`/cache/hf`) at
campaign init and reused read-only by every orbit. No API calls, no usage
fees, no rate limits.

**Orbit-side LLM use (optional, not required).** Some orbit search
strategies (LLM-proposal-based MAP-Elites, tree-of-thought θ proposals) may
want an LLM. Orbits that do are responsible for mounting their own Modal
secret and declaring it in their container image. Recommended paths:

- `anthropic-api-key` Modal secret  → Anthropic SDK
- `openai-api-key` Modal secret      → OpenAI SDK
- Local OPENROUTER_API_KEY + mount  → OpenRouter (cheapest for bulk)

Evaluator does NOT require or mount any of these.

## Chosen approach

- **Provider:** none (evaluator is self-contained; FMs are local).
- **Modal image:** `jax[cuda12]`, `flax==0.10.2`, `transformers==4.47.1`,
  `evosax==0.1.6`, `qdax==0.5.1`, `torch` (SigLIP PyTorch-subprocess path),
  `pyyaml`, `imageio[ffmpeg]` (mp4 encoding).
- **Secret mounts at eval time:** none.
- **HF cache:** Modal Volume `hf-cache` mounted at `/cache/hf`, populated
  once via bootstrap script.
- **JAX JIT cache:** Modal Volume `jax-jit-cache` mounted at `/cache/jax`,
  `JAX_COMPILATION_CACHE_DIR=/cache/jax` in container env.

Committed at Phase 2.0a; all Phase 2.2 eval-designer candidates should
converge on this approach.
