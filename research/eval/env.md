# Evaluation Environment

## Modal secrets (available to evaluator.py via modal.Secret.from_name)

Relevant (non-Supabase-preview):
- **`anthropic-api-key`**       — ANTHROPIC_API_KEY (**REQUIRED by evaluator for judge**)
- `openai-api-key`              — OPENAI_API_KEY (reserved for optional 2nd-judge ensembling)
- `canele-github`               — GitHub token (workspace)
- `canele-secrets`              — workspace secret bag
- `canele-eln-secrets`          — workspace secret bag
- `adam-secrets`, `mochi-secrets`, `scimarket-secrets` — workspace secrets (unused)

No `huggingface` secret — and none is needed (HF weights for CLIP are public).

## Local shell env vars

- ANTHROPIC_API_KEY: missing
- OPENAI_API_KEY: missing
- OPENROUTER_API_KEY: **present**
- GEMINI_API_KEY: missing
- GOOGLE_API_KEY: missing
- HF_TOKEN: missing

Local shell vars are not used by the Modal evaluator. Kept for reference.

## VLM / LLM provider — chosen approach

### Judge (final scoring)

- **Provider:** Anthropic
- **Model:** `claude-sonnet-4-6` (pinned; upgrades bump eval-vN)
- **SDK:** `anthropic` Python SDK
- **Secret mount:** `modal.Secret.from_name("anthropic-api-key")` into
  judge container
- **Container:** CPU-only (judge makes API calls, doesn't need GPU)
- **Concurrency:** 8 parallel calls via `asyncio.gather`
- **Rubric:** frozen in `research/eval/judge/rubric.md`, SHA-hashed at
  evaluator startup

### Search-time inner-loop signal (orbit's CLIP use)

- **Provider:** local (no API calls)
- **Models:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`)
- **SDK:** `transformers` `FlaxCLIPModel`
- **Weights:** public HF download, cached in Modal Volume `hf-cache` at
  `/cache/hf`
- **Container:** A100 GPU, network egress to `api.anthropic.com` and
  `api.openai.com` blocked (orbits cannot bypass the judge isolation)

### Previously considered, now dropped

- DINOv2 + SigLIP ensemble at scoring: dropped in favor of single-judge.
  CLIP still available inside `search()` as the orbit's inner-loop proxy.
- Temporal FMs (V-JEPA, VideoMAE, InternVideo2): deferred to campaign 2.
- Prompt bank at scoring: dropped — rubric replaces it.

## Gate check

- ✅ Anthropic Modal secret present: `anthropic-api-key`
- ✅ CLIP weights pullable from HF without auth
- ✅ Modal workspace has GPU access (A100 verified at Phase 0.6)
- ✅ No gated-model dependencies
- ⚠ Must populate `/cache/hf` with CLIP weights once at campaign
  bootstrap (pre-Phase 2.5)
