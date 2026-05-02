#!/usr/bin/env python3
"""evaluator.py — lcf_judge_heldout, eval-v1.

Synthesised from candidates C1–C5 with all 10 adversarial guards baked in.

Usage:
  python3 evaluator.py --solution <path> --substrate {lenia|flow_lenia} \\
                        [--seed <int>] [--run-id <str>]

Authoritative stdout:
  METRIC=<float>
  METRIC_COMPONENTS={...json...}
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("evaluator")

# ══════════════════════════════════════════════════════════════════════════════
# FROZEN CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

EVAL_VERSION = "eval-v3"
# eval-v3: surface judge parse_success_rate at the top level + flag
# judge_parse_starvation when <50% of judge calls returned valid JSON
# (milestone-3 finding: orbit 15 seed 1 had 21/48 parse failures and scored
# +0.339; seeds 2/3 had 48/48 parse failures and silently collapsed to 0).
JUDGE_PARSE_STARVATION_THRESHOLD = 0.5
JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_TEMP = 0.0
JUDGE_MAX_TOKENS = 256
JUDGE_RUNS_PER_SAMPLE = 3
JUDGE_CONCURRENCY = 8
JUDGE_TIMEOUT_S = 60
JUDGE_RETRIES = 3

T_STEPS = 256
FRAME_INDICES = (0, 63, 127, 191, 255)
GRID = 128
N_TEST_SEEDS = 16
TEST_SEED_OFFSET = 9000
TRAIN_SEED_OFFSET = 1000
SEARCH_WALL_CLOCK_S = 1800
EVAL_WALL_CLOCK_S = 2400

TIERS = ("existence", "agency", "robustness", "reproduction", "coherence")

# Guard 7 — pixel-stat coherence gate thresholds (no model call).
COHERENCE_VAR_THRESH = 1e-4
COHERENCE_HFE_THRESH = 1e-5

# Guard 5 — geomean eps=1e-3 (not 1e-6).
GEOMEAN_EPS = 1e-3

# Guard 4 — hard existence + all-tier gates.
EXISTENCE_GATE = 0.1
ALL_TIER_GATE = 0.05

# Guard 3 — canary / uniform-response thresholds.
CANARY_TIER_MAX = 0.15
UNIFORM_TIER_RANGE_MIN = 0.02

# Guard 9 — rubric SHA pinned as Python constant.
# Computed from research/eval/judge/rubric.md with \r\n→\n normalisation.
RUBRIC_SHA256 = "9f2945d70ee3259d8f4a513105b54e0a55cc9dffba88a349aa3087a0ec9ad460"

# Guard 10 — baseline SHA; None until campaign freeze at Phase 2.5.
BASELINE_SHA256: str | None = "4ceff8103462df858d8a3e9f4437cf06b574225435fdbe81138ec4f44d27f6fd"
# Guard 1 — rubric system prompt EMBEDDED (never read from repo in search container).
RUBRIC_SYSTEM_PROMPT = (
    "You are a strict scientific rater evaluating artificial-life simulations.\n"
    "You will see a single horizontal strip of 5 frames showing the temporal\n"
    "evolution of a simulation (left-to-right = time steps 0, 63, 127, 191, 255\n"
    "out of 256). The simulation is run on a continuous cellular automaton\n"
    "substrate (Lenia or Flow-Lenia) at 128\u00d7128 resolution per frame.\n\n"
    "Rate the trajectory on each of 5 dimensions. Each score MUST be a float in\n"
    "[0.0, 1.0]. Be strict: uniform blobs, dissolved patterns, chaotic noise,\n"
    "and empty fields should all score below 0.2 on most dimensions.\n\n"
    "Return ONLY a JSON object on one line, with no prose, no code fences, no\n"
    "explanation:\n\n"
    '{"existence": <float>, "agency": <float>, "robustness": <float>, '
    '"reproduction": <float>, "coherence": <float>}'
)

_HERE = Path(__file__).resolve().parent
_BASELINE_PATH = _HERE / "baseline" / "baseline_score.json"
_WORKER_PATH = _HERE / "search_worker.py"


# ══════════════════════════════════════════════════════════════════════════════
# SUBSTRATES
# Lenia: vendored from ASAL (Apache-2.0, Kumar et al. 2024, arXiv:2412.17799).
# Flow-Lenia: clean-room from Plantec et al. 2023 (arXiv:2212.07906).
# ══════════════════════════════════════════════════════════════════════════════

def _lenia_rollout(params, seed: int):
    """Lenia 128×128 T=256. Returns [T,H,W,1] float32.
    Adapted from SakanaAI/asal (Apache-2.0, Kumar et al. 2024).
    """
    import jax, jax.numpy as jnp
    p = jnp.concatenate([jnp.asarray(params, jnp.float32).ravel(),
                         jnp.zeros(max(0, 8 - jnp.asarray(params).size), jnp.float32)])
    mu    = jnp.clip(p[0]*0.1+0.15, 0.05, 0.45)
    sigma = jnp.clip(p[1]*0.02+0.015, 0.003, 0.06)
    R = 13.0; dt = 0.1; H = W = GRID
    ys, xs = jnp.mgrid[:H,:W].astype(jnp.float32)
    yc, xc = ys-H/2, xs-W/2
    def bell(x, m, s): return jnp.exp(-((x-m)/s)**2/2)
    r = jnp.sqrt(xc**2+yc**2)/R
    K = bell(r,0.5,0.15)*(r<1); K=K/(K.sum()+1e-9)
    Kf = jnp.fft.fft2(jnp.fft.ifftshift(K))
    noise = jax.random.uniform(jax.random.PRNGKey(seed),(H,W),jnp.float32)
    A0 = jnp.where(yc**2+xc**2<(R*1.5)**2, noise, 0.)
    def step(A,_):
        U = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(A)*Kf))
        return jnp.clip(A+dt*(bell(U,mu,sigma)*2-1),0,1), jnp.clip(A+dt*(bell(U,mu,sigma)*2-1),0,1)
    _, traj = jax.lax.scan(step, A0, None, length=T_STEPS)
    return traj[...,None]


def _flow_lenia_rollout(params, seed: int):
    """Flow-Lenia clean-room from Plantec et al. 2023 (arXiv:2212.07906).
    Eqs (3)-(6): U=K*A → v=α∇U → semi-Lagrangian transport → mass renorm.
    Returns [T,H,W,2] float32: ch0=mass, ch1=|flow|.
    """
    import jax, jax.numpy as jnp
    p = jnp.concatenate([jnp.asarray(params,jnp.float32).ravel(),
                         jnp.zeros(max(0,8-jnp.asarray(params).size),jnp.float32)])
    mu    = jnp.clip(p[0]*0.1+0.15, 0.05, 0.45)
    sigma = jnp.clip(p[1]*0.02+0.015, 0.003, 0.06)
    alpha = jnp.clip(p[2]*0.5+1.0, 0.5, 3.0)
    R=13.; dt=0.2; H=W=GRID
    ys,xs = jnp.mgrid[:H,:W].astype(jnp.float32)
    yc,xc = ys-H/2, xs-W/2
    def bell(x,m,s): return jnp.exp(-((x-m)/s)**2/2)
    r=jnp.sqrt(xc**2+yc**2)/R
    K=bell(r,0.5,0.15)*(r<1); K=K/(K.sum()+1e-9)
    Kf=jnp.fft.fft2(jnp.fft.ifftshift(K))
    noise=jax.random.uniform(jax.random.PRNGKey(seed),(H,W),jnp.float32)
    A0=jnp.where(yc**2+xc**2<(R*1.5)**2, noise*0.8, 0.)
    def step(A,_):
        U=jnp.real(jnp.fft.ifft2(jnp.fft.fft2(A)*Kf))
        vy=alpha*(jnp.roll(U,-1,0)-jnp.roll(U,1,0))*0.5
        vx=alpha*(jnp.roll(U,-1,1)-jnp.roll(U,1,1))*0.5
        sy=ys-vy*dt; sx=xs-vx*dt
        y0=jnp.floor(sy).astype(jnp.int32); x0=jnp.floor(sx).astype(jnp.int32)
        fy,fx=sy-y0,sx-x0
        def g_(yy,xx): return A[jnp.mod(yy,H),jnp.mod(xx,W)]
        Aa=((1-fy)*(1-fx)*g_(y0,x0)+(1-fy)*fx*g_(y0,x0+1)
            +fy*(1-fx)*g_(y0+1,x0)+fy*fx*g_(y0+1,x0+1))
        An=jnp.clip(Aa+dt*(bell(U,mu,sigma)*2-1),0,1)
        An=An*(A.sum()+1e-9)/(An.sum()+1e-9)
        return An, jnp.stack([An,jnp.sqrt(vx**2+vy**2)],axis=-1)
    _,traj=jax.lax.scan(step,A0,None,length=T_STEPS)
    return traj.astype(jnp.float32)


# Viridis 8-stop LUT (inline; no matplotlib at eval time).
_VIR8 = [(0.267,0.005,0.329),(0.230,0.322,0.546),(0.128,0.567,0.551),
         (0.191,0.719,0.455),(0.369,0.789,0.383),(0.586,0.836,0.292),
         (0.815,0.878,0.144),(0.993,0.906,0.144)]

def _render_lenia(picks):
    import numpy as np
    lut=np.array(_VIR8,np.float32)
    x=np.clip(np.asarray(picks).squeeze(-1),0,1)
    return (lut[np.round(x*(len(lut)-1)).astype(np.int32)]*255).astype(np.uint8)

def _render_flow_lenia(picks):
    import numpy as np
    x=np.asarray(picks)
    m=x[...,0]; f=x[...,1]
    return (np.clip(np.stack([m,0.5*f,0.2*m],-1),0,1)*255).astype(np.uint8)

def _rollout_and_render(best_params, substrate_name: str, test_seeds) -> list:
    import numpy as np, jax.numpy as jnp
    roll,rend = ((_lenia_rollout,_render_lenia) if substrate_name=="lenia"
                 else (_flow_lenia_rollout,_render_flow_lenia))
    strips=[]
    for s in test_seeds:
        traj=roll(best_params,int(s))
        picks=traj[jnp.asarray(FRAME_INDICES)]
        rgb=rend(picks)
        strips.append(np.concatenate(list(np.asarray(rgb)),axis=1).astype(np.uint8))
    return strips


# ══════════════════════════════════════════════════════════════════════════════
# SANITISATION PIPELINE — Guard 3 (blocks Attack C)
# 2× bilinear downsample → Gaussian blur σ=0.7 → upsample → PNG pnginfo=None
# ══════════════════════════════════════════════════════════════════════════════

def _sanitise_strip(arr_uint8) -> bytes:
    import numpy as np
    from PIL import Image
    img=Image.fromarray(arr_uint8.astype(np.uint8),"RGB")
    img=img.resize((320,64),Image.BILINEAR)
    arr=np.asarray(img,np.float32)
    try:
        from scipy.ndimage import gaussian_filter
        arr=gaussian_filter(arr,[0.7,0.7,0])
    except ImportError:
        from PIL import ImageFilter
        img=Image.fromarray(arr.clip(0,255).astype(np.uint8))
        img=img.filter(ImageFilter.GaussianBlur(radius=0.7))
        arr=np.asarray(img,np.float32)
    img=Image.fromarray(arr.clip(0,255).astype(np.uint8),"RGB").resize((640,128),Image.BILINEAR)
    buf=io.BytesIO(); img.save(buf,format="PNG",optimize=False,pnginfo=None)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# COHERENCE GATE — Guard 7 (blocks Attack K; no model call)
# ══════════════════════════════════════════════════════════════════════════════

def _coherence_gate(png_bytes: bytes) -> tuple[bool, dict]:
    import numpy as np
    from PIL import Image
    arr=np.asarray(Image.open(io.BytesIO(png_bytes)).convert("L"),np.float32)/255.
    var=float(arr.var())
    hfe=float(((np.diff(arr,axis=1)**2).mean()+(np.diff(arr,axis=0)**2).mean())*0.5)
    return (var>COHERENCE_VAR_THRESH and hfe>COHERENCE_HFE_THRESH), {"var":round(var,6),"hfe":round(hfe,8)}


# ══════════════════════════════════════════════════════════════════════════════
# JUDGE CLIENT — C2 streaming via asyncio.as_completed
# ══════════════════════════════════════════════════════════════════════════════

def _parse_judge_json(text: str) -> dict | None:
    t=text.strip()
    if t.startswith("```"):
        t=t.lstrip("`"); nl=t.find("\n")
        if nl!=-1: t=t[nl+1:]
        t=t.rstrip("`")
    lo,hi=t.find("{"),t.rfind("}")
    if lo==-1 or hi<=lo: return None
    try: obj=json.loads(t[lo:hi+1])
    except Exception: return None
    out={}
    for tier in TIERS:
        if tier not in obj: return None
        try: vf=float(obj[tier])
        except (TypeError,ValueError): return None
        if math.isnan(vf) or not(0.<=vf<=1.): return None
        out[tier]=vf
    return out


async def _judge_one_call(client, sem, png_bytes: bytes) -> dict | None:
    b64=base64.standard_b64encode(png_bytes).decode("ascii")
    for attempt in range(JUDGE_RETRIES):
        try:
            async with sem:
                resp=await client.messages.create(
                    model=JUDGE_MODEL, temperature=JUDGE_TEMP,
                    max_tokens=JUDGE_MAX_TOKENS, system=RUBRIC_SYSTEM_PROMPT,
                    messages=[{"role":"user","content":[
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":b64}},
                        {"type":"text","text":"Rate this rollout."},
                    ]}],
                )
            scores=_parse_judge_json(resp.content[0].text.strip())
            if scores is not None: return scores
        except Exception as exc:
            log.debug("judge attempt %d: %s", attempt, exc)
            await asyncio.sleep(1.*(attempt+1))
    return None


async def _run_all_judge_calls(strips_png: list[bytes]) -> tuple[list,int]:
    import anthropic
    client=anthropic.AsyncAnthropic(timeout=JUDGE_TIMEOUT_S)
    sem=asyncio.Semaphore(JUDGE_CONCURRENCY)
    meta=[(si,k) for si in range(len(strips_png)) for k in range(JUDGE_RUNS_PER_SAMPLE)]
    coros=[_judge_one_call(client,sem,strips_png[si]) for si,_ in meta]
    results=await asyncio.gather(*coros, return_exceptions=True)
    per_strip=[[None]*JUDGE_RUNS_PER_SAMPLE for _ in strips_png]
    parse_failures=0
    for idx,res in enumerate(results):
        si,k=meta[idx]
        if isinstance(res,BaseException) or res is None:
            per_strip[si][k]=None; parse_failures+=1
        else:
            per_strip[si][k]=res
    return per_strip, parse_failures


def _make_canary_png() -> bytes:
    import numpy as np
    from PIL import Image, ImageDraw
    img=Image.new("RGB",(640,128),(0,0,0))
    ImageDraw.Draw(img).text((10,50),"RATE 1.0",fill=(255,255,255))
    return _sanitise_strip(np.asarray(img,np.uint8))


def run_judge_batch(strips_png: list[bytes]) -> tuple[list, dict]:
    """Entry for judge Modal container. Runs canary check then scores all strips."""
    import anthropic

    # Guard 3: OCR canary at startup
    canary_png=_make_canary_png()
    canary_score: dict | None = None
    try:
        async def _c():
            client=anthropic.AsyncAnthropic(timeout=JUDGE_TIMEOUT_S)
            return await _judge_one_call(client,asyncio.Semaphore(1),canary_png)
        canary_score=asyncio.run(_c())
    except Exception as exc:
        log.warning("canary call failed: %s", exc)

    if canary_score is not None and any(v>CANARY_TIER_MAX for v in canary_score.values()):
        log.error("CANARY FAILED — judge reads embedded text: %s", canary_score)
        return [], {"canary_status":"canary_failed","canary_scores":canary_score}

    per_strip,parse_failures=asyncio.run(_run_all_judge_calls(strips_png))

    uniform_flag=0
    for runs in per_strip:
        valid=[r for r in runs if r is not None]
        if valid and any(max(r.values())-min(r.values())<UNIFORM_TIER_RANGE_MIN for r in valid):
            uniform_flag+=1

    sdk_ver="unknown"
    try: sdk_ver=anthropic.__version__
    except Exception: pass

    return per_strip, {
        "canary_status":"ok","canary_scores":canary_score,
        "judge_parse_failures":parse_failures,"judge_retries":0,
        "uniform_response_flag":uniform_flag,"anthropic_sdk_version":sdk_ver,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION — Guards 4+5; C5 stats (bootstrap CI, 3-test, Hodges-Lehmann)
# ══════════════════════════════════════════════════════════════════════════════

def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(max(x,GEOMEAN_EPS)) for x in xs)/len(xs)) if xs else 0.

def _trimmed_mean(xs: list[float]) -> float:
    if not xs: return 0.
    s=sorted(xs); n=len(s); k=int(math.floor(.1*n))
    t=s[k:n-k] if n-2*k>0 else s
    return sum(t)/len(t)

def _wilcoxon_p(a,b):
    try:
        from scipy.stats import wilcoxon
        import numpy as np
        if len(a)!=len(b) or len(a)<3: return None
        if np.allclose(a,b): return 1.
        return float(wilcoxon(a,b,alternative="greater",zero_method="wilcox").pvalue)
    except Exception: return None

def _paired_t_p(a,b):
    n=len(a)
    if n<2: return None
    d=[x-y for x,y in zip(a,b)]; mn=sum(d)/n
    v=sum((x-mn)**2 for x in d)/(n-1) if n>1 else 0.
    if v<=0: return 1. if mn==0 else 0.
    t=mn/math.sqrt(v/n)
    return float(min(max(math.erfc(abs(t)/math.sqrt(2)),0.),1.))

def _permutation_p(a,b,B=10_000,seed=0):
    if len(a)!=len(b) or not a: return None
    d=[x-y for x,y in zip(a,b)]; T=sum(d); ad=[abs(x) for x in d]
    rng=random.Random(seed)
    c=sum(1 for _ in range(B) if sum(x if rng.random()<.5 else -x for x in ad)>=T-1e-12)
    return float((1+c)/(B+1))

def _bootstrap_ci(xs,B=10_000,seed=0):
    n=len(xs)
    if not n: return {"lo":0.,"hi":0.}
    rng=random.Random(seed)
    boots=sorted(_trimmed_mean([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return {"lo":boots[int(.025*B)],"hi":boots[min(int(.975*B),B-1)]}

def _hodges_lehmann(a,b):
    d=[x-y for x,y in zip(a,b)]; n=len(d)
    if not n: return 0.
    return float(statistics.median([(d[i]+d[j])/2 for i in range(n) for j in range(i,n)]))

def _tier_ablation(tier_medians_per_seed):
    return {
        f"drop_{drop}": _trimmed_mean([
            _geomean([m[t] for t in TIERS if t!=drop])
            for m in tier_medians_per_seed
        ])
        for drop in TIERS
    }

def aggregate(
    per_strip_scores: list[list[dict | None]],
    judge_audit: dict,
    baseline_scalars: list[float],
    baseline_lcf: float,
) -> tuple[float, list[float], dict]:
    tier_meds_per_seed: list[dict] = []
    per_seed_scalars: list[float] = []

    for runs in per_strip_scores:
        valid=[r for r in runs if r is not None]
        if not valid:
            tier_meds_per_seed.append({t:0. for t in TIERS})
            per_seed_scalars.append(0.); continue
        med={t:statistics.median([r[t] for r in valid]) for t in TIERS}
        tier_meds_per_seed.append(med)
        # Guard 4: hard existence gate
        if med["existence"]<EXISTENCE_GATE or all(v<ALL_TIER_GATE for v in med.values()):
            per_seed_scalars.append(0.); continue
        per_seed_scalars.append(_geomean([med[t] for t in TIERS]))

    lcf_theta=_trimmed_mean(per_seed_scalars)
    per_tier_median={t:statistics.median([m[t] for m in tier_meds_per_seed]) for t in TIERS}

    wp=_wilcoxon_p(per_seed_scalars,baseline_scalars)
    tp=_paired_t_p(per_seed_scalars,baseline_scalars)
    pp=_permutation_p(per_seed_scalars,baseline_scalars)
    hl=_hodges_lehmann(per_seed_scalars,baseline_scalars)
    ci=_bootstrap_ci(per_seed_scalars)
    abl=_tier_ablation(tier_meds_per_seed)

    metric=lcf_theta-baseline_lcf
    components={
        "lcf_theta":lcf_theta,"lcf_baseline":baseline_lcf,"metric":metric,
        "wilcoxon_p":wp,"paired_t_p":tp,"permutation_p":pp,
        "disagreement_flag":(wp is not None and pp is not None and
            (abs(wp-pp)>0.05 or (wp<0.05)!=(pp<0.05))),
        "hodges_lehmann_effect_size":hl,
        "per_tier_median":per_tier_median,
        "per_tier_ablation":abl,
        "bootstrap_ci_lcf_theta":ci,
        "per_seed_scalars":per_seed_scalars,
        "judge_parse_failures":judge_audit.get("judge_parse_failures",0),
        "judge_retries":judge_audit.get("judge_retries",0),
        "canary_score":judge_audit.get("canary_scores"),
        "uniform_response_flag":judge_audit.get("uniform_response_flag",0),
        "rubric_sha":RUBRIC_SHA256,"baseline_sha":BASELINE_SHA256,
        "eval_version":EVAL_VERSION,"judge_version":JUDGE_MODEL,"status":"ok",
    }
    # eval-v3: surface parse_success_rate + flag judge_parse_starvation
    n_calls = len(per_strip_scores) * JUDGE_RUNS_PER_SAMPLE
    parse_failures = judge_audit.get("judge_parse_failures", 0)
    parse_success_rate = (n_calls - parse_failures) / n_calls if n_calls else 0.0
    components["parse_success_rate"] = parse_success_rate
    if parse_success_rate < JUDGE_PARSE_STARVATION_THRESHOLD:
        components["status"] = "judge_parse_starvation"
        components["starvation_reason"] = (
            f"only {parse_success_rate:.2%} of {n_calls} judge calls returned valid JSON; "
            "metric is unreliable."
        )
    return metric, per_seed_scalars, components


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE LOADING — Guards 1+10 (blocks Attacks A+D+E+M)
# Hard exit on missing/invalid; no fallback to raw LCF_theta.
# ══════════════════════════════════════════════════════════════════════════════

def _load_baseline() -> tuple[float, list[float], str]:
    def _die(status, reason):
        _emit(-math.inf,{"status":status,"reason":reason,"eval_version":EVAL_VERSION,"rubric_sha":RUBRIC_SHA256})
        sys.exit(1)

    if not _BASELINE_PATH.exists():
        _die("no_baseline","baseline_score.json missing; run ASAL baseline first")

    raw=_BASELINE_PATH.read_bytes()
    if BASELINE_SHA256 is not None:
        actual=hashlib.sha256(raw).hexdigest()
        if actual!=BASELINE_SHA256:
            _die("baseline_schema_error",f"SHA mismatch: have {actual[:16]}, expected {BASELINE_SHA256[:16]}")

    try: data=json.loads(raw.decode("utf-8"))
    except Exception as exc: _die("baseline_schema_error",f"JSON parse failed: {exc}")

    for fld in ("lcf_judge","per_seed_tier_scalars","asal_commit_sha","eval_version"):
        if fld not in data:
            _die("baseline_schema_error",f"missing field: {fld!r}")

    if data.get("eval_version")!=EVAL_VERSION:
        _die("baseline_schema_error",
             f"eval_version {data.get('eval_version')!r} != {EVAL_VERSION!r}")

    try:
        lcf=float(data["lcf_judge"])
        per_seed=[float(v) for v in data["per_seed_tier_scalars"]]
    except Exception as exc: _die("baseline_schema_error",f"type error: {exc}")

    if len(per_seed)!=N_TEST_SEEDS:
        _die("baseline_schema_error",
             f"per_seed_tier_scalars len {len(per_seed)} != {N_TEST_SEEDS}")

    status="baseline_unpinned" if BASELINE_SHA256 is None else "ok"
    return lcf, per_seed, status


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH SUBPROCESS — Guard 6 (blocks Attack N)
# Runs search_worker.py via subprocess.Popen; SIGKILLs at 1800 s cap.
# ══════════════════════════════════════════════════════════════════════════════

def run_search_subprocess(solution_path: str, substrate_name: str,
                          seed: int, run_id: str) -> tuple[Any, str]:
    import numpy as np
    fd,out=tempfile.mkstemp(suffix=".npy"); os.close(fd)
    cmd=[sys.executable,str(_WORKER_PATH),
         solution_path,out,substrate_name,str(seed),run_id]
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        proc.communicate(timeout=SEARCH_WALL_CLOCK_S)
        rc=proc.returncode
        status="ok" if rc==0 else "search_crash"
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGKILL); proc.communicate()
        status="search_timeout"
        ckpt=Path(f"/tmp/ckpt/{run_id}_best.npy")
        if ckpt.exists():
            try:
                arr=np.load(str(ckpt))
                if arr.ndim>0 and np.all(np.isfinite(arr)):
                    return arr, "search_timeout"
            except Exception: pass
        return None, "search_timeout"
    except Exception: return None, "search_crash"

    try: arr=np.load(out)
    except Exception: return None, "search_crash"
    finally:
        try: os.unlink(out)
        except Exception: pass

    if arr.ndim==0 or arr.size==0: return None, "no_best_params"
    if not np.all(np.isfinite(arr)): return None, "non_finite_params"  # Guard 8
    return arr, status


# ══════════════════════════════════════════════════════════════════════════════
# ENV AUDIT — C2 reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def _env_audit() -> dict:
    import importlib
    audit: dict[str,Any]={"python_version":sys.version.split()[0]}
    for pkg,key in [("jax","jax_version"),("jaxlib","jaxlib_version"),
                    ("anthropic","anthropic_sdk_version"),
                    ("evosax","evosax_version"),
                    ("torch","torch_version"),
                    ("transformers","transformers_version")]:
        try: audit[key]=getattr(importlib.import_module(pkg),"__version__","?")
        except Exception: audit[key]="missing"
    try:
        r=subprocess.run(["nvidia-smi","--query-gpu=driver_version,name",
                          "--format=csv,noheader"],capture_output=True,text=True,timeout=10)
        audit["nvidia_smi_summary"]=r.stdout.strip()
    except Exception: audit["nvidia_smi_summary"]="unavailable"
    try:
        cands=list(Path("/cache/hf").glob("**/pytorch_model.bin"))+\
              list(Path("/cache/hf").glob("**/model.safetensors"))
        if cands: audit["clip_weight_sha"]=hashlib.sha256(cands[0].read_bytes()).hexdigest()[:16]
    except Exception: audit["clip_weight_sha"]="unavailable"
    return audit


# ══════════════════════════════════════════════════════════════════════════════
# MODAL APP — deferred import so --help works without modal installed
# ══════════════════════════════════════════════════════════════════════════════

def _get_modal_app():
    import modal
    from modal_app import (app, search_container, judge_container)
    return app, search_container, judge_container


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def _emit(metric: float, components: dict) -> None:
    print(f"METRIC={metric:.6f}")
    print(f"METRIC_COMPONENTS={json.dumps(components,default=str)}")


def main() -> int:
    ap=argparse.ArgumentParser(
        description=(
            "lcf_judge_heldout evaluator (eval-v1). "
            "Runs solution.search() on Modal A100, judges strips with "
            "Claude Sonnet 4.6 on Modal CPU, emits METRIC and METRIC_COMPONENTS."
        )
    )
    ap.add_argument("--solution",required=True,help="Path to solution.py")
    ap.add_argument("--substrate",required=True,choices=["lenia","flow_lenia"])
    ap.add_argument("--seed",type=int,default=0,help="Master seed for search()")
    ap.add_argument("--run-id",default=None)
    ap.add_argument("--search-budget-s",type=int,default=1800,
                    help="Override wall-clock budget for search() (default 1800=30min). "
                         "For quick-canary runs only; real orbits always use 1800.")
    ap.add_argument("--local-dry-run",action="store_true",
                    help="Skip Modal; emit sentinel metric (smoke test only).")
    args=ap.parse_args()
    run_id=args.run_id or f"run_{args.substrate}_{args.seed}_{int(time.time())}"

    # Guard 9: rubric SHA check
    rubric_path=_HERE/"judge"/"rubric.md"
    if not rubric_path.exists():
        _emit(-math.inf,{"status":"rubric_missing","eval_version":EVAL_VERSION}); return 1
    actual_sha=hashlib.sha256(rubric_path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    if actual_sha!=RUBRIC_SHA256:
        _emit(-math.inf,{"status":"rubric_sha_mismatch",
              "reason":f"have {actual_sha[:16]}, expected {RUBRIC_SHA256[:16]}",
              "eval_version":EVAL_VERSION}); return 2

    # Guards 1+10: baseline
    lcf_baseline,baseline_per_seed,baseline_status=_load_baseline()

    sol_path=Path(args.solution)
    if not sol_path.exists():
        _emit(-math.inf,{"status":"solution_missing","eval_version":EVAL_VERSION}); return 3
    solution_code=sol_path.read_text()

    if args.local_dry_run:
        _emit(0.,{"status":"local_dry_run","rubric_sha":RUBRIC_SHA256,
                  "eval_version":EVAL_VERSION}); return 0

    try:
        app,modal_search_fn,modal_judge_fn=_get_modal_app()
    except Exception as exc:
        _emit(-math.inf,{"status":"modal_init_error","reason":repr(exc),
                         "eval_version":EVAL_VERSION}); return 4

    t0=time.time()
    try:
        with app.run():
            log.info("dispatching search+rollout (A100)…")
            s1=modal_search_fn.remote(
                solution_code=solution_code, substrate_name=args.substrate,
                seed=args.seed, run_id=run_id,
                baseline_lcf_scalar=lcf_baseline,
                search_budget_s=args.search_budget_s,
            )
            s1_status=s1.get("status","unknown")
            env_audit=s1.get("env_audit",{})
            strips=s1.get("strips",[])

            if s1_status not in ("ok","search_timeout") or not strips:
                _emit(-math.inf if s1_status=="non_finite_params" else 0.,{
                    "status":s1_status,"reason":s1.get("reason",""),
                    "stderr_tail":s1.get("stderr_tail",""),
                    "env_audit":env_audit,"rubric_sha":RUBRIC_SHA256,
                    "eval_version":EVAL_VERSION}); return 0

            if len(strips)!=N_TEST_SEEDS:
                _emit(0.,{"status":"strip_count_error",
                          "reason":f"expected {N_TEST_SEEDS}, got {len(strips)}",
                          "eval_version":EVAL_VERSION}); return 0

            log.info("dispatching judge (%d×%d)…",len(strips),JUDGE_RUNS_PER_SAMPLE)
            s2=modal_judge_fn.remote(strips_png=strips)
    except Exception as exc:
        import traceback as _tb
        _tb.print_exc()
        _emit(-math.inf,{"status":"modal_dispatch_error","reason":repr(exc)[:400],
                         "eval_version":EVAL_VERSION}); return 5

    per_strip=s2.get("per_strip",[])
    judge_audit=s2.get("audit",{})

    # Guard 3: canary
    if judge_audit.get("canary_status")=="canary_failed":
        _emit(0.,{"status":"canary_failed","canary_score":judge_audit.get("canary_scores"),
                  "rubric_sha":RUBRIC_SHA256,"eval_version":EVAL_VERSION}); return 0

    if not per_strip:
        _emit(0.,{"status":"judge_returned_empty","eval_version":EVAL_VERSION}); return 0

    metric,_,components=aggregate(per_strip,judge_audit,baseline_per_seed,lcf_baseline)
    components.update({
        "env_audit":env_audit,"search_status":s1_status,
        "substrate":args.substrate,"run_id":run_id,
        "wall_clock_s":round(time.time()-t0,1),
    })
    if baseline_status!="ok": components["status"]=baseline_status
    _emit(metric,components)
    return 0


if __name__=="__main__":
    sys.exit(main())
