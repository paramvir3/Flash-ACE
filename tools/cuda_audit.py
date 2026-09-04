#!/usr/bin/env python3
"""CUDA readiness audit for TRACE-v2.

Verifies correctness on GPU and measures each of the ranked performance gaps.
Run from the repository root:

    python tools/cuda_audit.py                 # everything
    python tools/cuda_audit.py --quick         # correctness only
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from e3nn import o3

from flashace.model import TransformersACE

CKPT = "training/model.pt"
FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")
    if not ok:
        FAILURES.append(name)


def load(dtype=torch.float32, device="cuda"):
    ck = torch.load(CKPT, map_location="cpu")
    cfg = ck["config"]
    keys = [
        "r_max", "l_max", "num_radial", "hidden_dim", "num_layers",
        "correlation_order", "correlation_channels", "radial_mlp_hidden",
        "radial_mlp_layers", "attention_num_heads", "attention_key_dim",
        "attention_ffn_hidden", "attention_dropout", "attention_layer_scale_init",
        "attention_distance_penalty", "radial_basis_type", "radial_trainable",
        "gaussian_width",
    ]
    m = TransformersACE(**{k: cfg[k] for k in keys if k in cfg})
    m.load_state_dict(ck["model_state_dict"])
    return m.to(device=device, dtype=dtype).eval(), float(cfg["r_max"])


def data(atoms, rc, dtype, device):
    i, j, S = neighbor_list("ijS", atoms, rc)
    return {
        "z": torch.tensor(atoms.numbers, device=device),
        "pos": torch.tensor(atoms.positions, dtype=dtype, device=device),
        "cell": torch.tensor(atoms.cell.array, dtype=dtype, device=device),
        "edge_index": torch.stack([torch.tensor(j), torch.tensor(i)]).to(device),
        "edge_shift": torch.tensor(S, dtype=dtype, device=device),
        "volume": torch.tensor(float(atoms.get_volume()), device=device),
    }, len(i)


def bench(fn, device, n=10, warm=5):
    for _ in range(warm):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


# ----------------------------------------------------------------- correctness
def correctness(base, rc):
    print("\n=== CORRECTNESS ON CUDA ===")
    at = base.repeat((2, 2, 2))
    n = len(at)

    m32g, _ = load(torch.float32, "cuda")
    m32c, _ = load(torch.float32, "cpu")
    dg, nedge = data(at, rc, torch.float32, "cuda")
    dc, _ = data(at, rc, torch.float32, "cpu")
    print(f"  system: {n} atoms, {nedge} edges")

    Eg, Fg, Sg, _ = m32g(dg, training=False, compute_stress=True)
    Ec, Fc, Sc, _ = m32c(dc, training=False, compute_stress=True)
    check("runs on CUDA (E + F + stress)", True, f"E = {float(Eg):.6f} eV")
    check("energy  CPU vs CUDA", abs(float(Eg) - float(Ec)) / n * 1000 < 1e-2,
          f"{abs(float(Eg)-float(Ec))/n*1000:.5f} meV/atom")
    check("forces  CPU vs CUDA", float((Fg.cpu() - Fc).abs().max()) < 5e-4,
          f"{float((Fg.cpu()-Fc).abs().max()):.3e} eV/A")
    check("stress  CPU vs CUDA", float((Sg.cpu() - Sc).abs().max()) < 1e-7,
          f"{float((Sg.cpu()-Sc).abs().max()):.3e} eV/A^3")

    m64, _ = load(torch.float64, "cuda")

    def run64(a2):
        d, _ = data(a2, rc, torch.float64, "cuda")
        return m64(d, training=False, compute_stress=False)

    E0, F0, _, _ = run64(at)
    we = wf = 0.0
    for _ in range(4):
        R = o3.rand_matrix(dtype=torch.float64).numpy()
        a2 = at.copy()
        a2.set_positions(at.positions @ R.T)
        a2.set_cell(at.cell.array @ R.T)
        E1, F1, _, _ = run64(a2)
        we = max(we, abs(float(E1) - float(E0)))
        wf = max(wf, float((F1.cpu() - F0.cpu() @ torch.tensor(R.T)).abs().max()))
    check("energy invariant under rotation", we / n * 1000 < 1e-6, f"{we:.2e} eV")
    check("forces equivariant under rotation", wf < 1e-8, f"{wf:.2e} eV/A")

    # finite differences: h must be large enough that cancellation does not
    # dominate. The error grows as h shrinks; 1e-3 is near the optimum here.
    p0 = at.positions.copy()

    def E_of(p):
        a2 = at.copy()
        a2.set_positions(p)
        return float(run64(a2)[0])

    h, err = 1e-3, 0.0
    for k in range(6):
        for a in range(3):
            pp = p0.copy(); pp[k, a] += h
            pm = p0.copy(); pm[k, a] -= h
            err = max(err, abs(-(E_of(pp) - E_of(pm)) / (2 * h) - float(F0[k, a].cpu())))
    check("F == -dE/dr (central FD, h=1e-3)", err < 1e-4, f"{err:.2e} eV/A")
    check("sum of forces == 0", float(F0.sum(0).abs().max()) < 1e-9,
          f"{float(F0.sum(0).abs().max()):.2e}")

    e1 = float(run64(at)[0]) / len(at)
    big = at.repeat((2, 1, 1))
    e2 = float(run64(big)[0]) / len(big)
    check("E/atom invariant to 2x1x1 replication", abs(e1 - e2) * 1000 < 1e-6,
          f"{abs(e1-e2)*1000:.2e} meV/atom")

    Ed, Fd, Sd, _ = m32g(dg, training=False, compute_stress=True)
    check("outputs stay on CUDA", Ed.is_cuda and Fd.is_cuda and Sd.is_cuda,
          f"{Ed.device}")
    return m32g, m64, at


# ------------------------------------------------------------------- the gaps
def gap1_batching(rc, base):
    print("\n=== GAP 1: batched graphs ===")
    from train import AtomisticDataset

    # Algebraic equivalence must be tested with deterministic kernels: CUDA's
    # index_add_ atomics reorder float additions, which alone accounts for
    # ~2e-9 relative in fp64 and would mask (or fake) a real discrepancy.
    torch.use_deterministic_algorithms(True, warn_only=True)
    m, _ = load(torch.float64, "cuda")
    frames = read("training/train.extxyz", index=":8")
    items = []
    for a in frames:
        i, j, S = neighbor_list("ijS", a, rc)
        items.append({
            "z": torch.tensor(a.numbers),
            "pos": torch.tensor(a.positions, dtype=torch.float64),
            "cell": torch.tensor(a.cell.array, dtype=torch.float64),
            "volume": torch.tensor(float(a.get_volume()), dtype=torch.float64),
            "edge_index": torch.stack([torch.tensor(j), torch.tensor(i)]),
            "edge_shift": torch.tensor(S, dtype=torch.float64),
            "t_E": torch.tensor(0.0, dtype=torch.float64),
            "t_F": torch.zeros((len(a), 3), dtype=torch.float64),
            "t_S": torch.zeros((3, 3), dtype=torch.float64),
            "has_stress": torch.tensor(True),
        })
    dev_items = [{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in it.items()}
                 for it in items]
    seqE, seqF, seqS = [], [], []
    for it in dev_items:
        E, F, S, _ = m(it, training=False, compute_stress=True)
        seqE.append(float(E)); seqF.append(F.detach()); seqS.append(S.detach())
    bat = AtomisticDataset.collate_batched(items)
    bat = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in bat.items()}
    bE, bF, bS, _ = m.forward_batched(bat, training=False, compute_stress=True)
    # dtype matters: torch.tensor([python floats]) defaults to float32 and would
    # inject ~1e-8 eV of truncation into a float64 comparison.
    ref_E = torch.tensor(seqE, dtype=bE.dtype)
    check("batched E == sequential E",
          float((bE.cpu() - ref_E).abs().max()) < 1e-12,
          f"{float((bE.cpu()-ref_E).abs().max()):.2e} eV")
    check("batched F == sequential F",
          float((bF - torch.cat(seqF)).abs().max()) < 1e-12,
          f"{float((bF-torch.cat(seqF)).abs().max()):.2e} eV/A")
    check("batched stress == sequential stress",
          float((bS - torch.stack(seqS)).abs().max()) < 1e-11,
          f"{float((bS-torch.stack(seqS)).abs().max()):.2e} eV/A^3")

    torch.use_deterministic_algorithms(False)

    m32, _ = load(torch.float32, "cuda")
    b32 = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v)
           for k, v in bat.items()}
    i32 = [{k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v)
            for k, v in it.items()} for it in dev_items]
    t_seq = bench(lambda: [m32(d, training=False, compute_stress=False) for d in i32], "cuda", 6, 3)
    t_bat = bench(lambda: m32.forward_batched(b32, training=False, compute_stress=False), "cuda", 6, 3)
    print(f"  8 structures sequential : {t_seq:8.2f} ms")
    print(f"  8 structures batched    : {t_bat:8.2f} ms    -> {t_seq/t_bat:.1f}x")


def gap2_compile(m32, dg):
    print("\n=== GAP 2: torch.compile ===")
    t_eager = bench(lambda: m32(dg, training=False, compute_stress=False), "cuda")
    try:
        mc = torch.compile(m32, dynamic=False)
        t_c = bench(lambda: mc(dg, training=False, compute_stress=False), "cuda", 6, 8)
        dE = abs(float(mc(dg, training=False, compute_stress=False)[0])
                 - float(m32(dg, training=False, compute_stress=False)[0]))
        print(f"  inference eager {t_eager:7.2f} ms | compiled {t_c:7.2f} ms "
              f"({t_eager/t_c:.2f}x) | dE {dE:.2e} eV")
    except Exception as e:
        print(f"  inference compile FAILED: {type(e).__name__}: {str(e)[:90]}")
    try:
        mt, _ = load(torch.float32, "cuda"); mt.train()
        mct = torch.compile(mt, dynamic=False)
        E, F, _, _ = mct(dg, training=True, compute_stress=False)
        ((E / 40) ** 2 + (F ** 2).mean()).backward()
        print("  training compile: WORKS")
    except Exception as e:
        print(f"  training compile FAILED (expected): {type(e).__name__}: {str(e)[:80]}")


def gap3_launches(m32, dg):
    print("\n=== GAP 3: kernel launches ===")
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            m32(dg, training=False, compute_stress=False)
        torch.cuda.synchronize()
    ka = prof.key_averages()
    n = sum(e.count for e in ka if e.self_device_time_total > 0) // 5
    gpu_ms = sum(e.self_device_time_total for e in ka) / 5 / 1000
    wall = bench(lambda: m32(dg, training=False, compute_stress=False), "cuda")
    print(f"  kernel launches / step : {n}")
    print(f"  GPU kernel time / step : {gpu_ms:7.2f} ms")
    print(f"  wall time / step       : {wall:7.2f} ms   -> GPU idle {100*(1-gpu_ms/wall):.0f}%")


def gap4_syncs(m32, dg):
    print("\n=== GAP 4: host synchronisation ===")
    def with_syncs():
        E, F, S, _ = m32(dg, training=False, compute_stress=True)
        _ = float(E); _ = F.pow(2).mean().item(); _ = S.pow(2).sum().item()
    def no_syncs():
        m32(dg, training=False, compute_stress=True)
    a = bench(with_syncs, "cuda"); b = bench(no_syncs, "cuda")
    print(f"  with .item() {a:7.2f} ms | without {b:7.2f} ms | delta {a-b:+.2f} ms ({100*(a-b)/a:+.1f}%)")


def gap5_determinism(rc, base):
    print("\n=== GAP 5: determinism ===")
    at = base.repeat((2, 2, 2))
    for tag, det in (("default", False), ("deterministic", True)):
        if det:
            torch.use_deterministic_algorithms(True, warn_only=True)
        for dt, lab in ((torch.float32, "fp32"), (torch.float64, "fp64")):
            m, _ = load(dt, "cuda")
            d, _ = data(at, rc, dt, "cuda")
            v = [float(m(d, training=False, compute_stress=False)[0]) for _ in range(8)]
            print(f"  {tag:<14}{lab}: {len(set(v))} distinct / 8   spread {max(v)-min(v):.3e} eV")
        if det:
            torch.use_deterministic_algorithms(False)
    m, _ = load(torch.float32, "cuda")
    d, _ = data(at, rc, torch.float32, "cuda")
    t0 = bench(lambda: m(d, training=False, compute_stress=False), "cuda")
    torch.use_deterministic_algorithms(True, warn_only=True)
    t1 = bench(lambda: m(d, training=False, compute_stress=False), "cuda")
    torch.use_deterministic_algorithms(False)
    print(f"  cost of deterministic mode: {t0:.2f} -> {t1:.2f} ms ({t1/t0:.2f}x)")


def gap6_tf32(rc, base):
    print("\n=== GAP 6: TF32 ===")
    at = base.repeat((3, 3, 2))
    m, _ = load(torch.float32, "cuda")
    d, _ = data(at, rc, torch.float32, "cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    t_off = bench(lambda: m(d, training=False, compute_stress=True), "cuda")
    E_off = float(m(d, training=False, compute_stress=False)[0])
    torch.backends.cuda.matmul.allow_tf32 = True
    t_on = bench(lambda: m(d, training=False, compute_stress=True), "cuda")
    E_on = float(m(d, training=False, compute_stress=False)[0])
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"  off {t_off:7.2f} ms | on {t_on:7.2f} ms ({t_off/t_on:.2f}x) | "
          f"dE {abs(E_on-E_off)/len(at)*1000:.4f} meV/atom")


def scaling(rc, base):
    print("\n=== SCALING (inference, E+F) ===")
    m, _ = load(torch.float32, "cuda")
    print(f"  {'atoms':>8}{'edges':>9}{'ms':>9}{'us/atom':>10}{'GPU mem':>10}")
    for r in [(2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 9, 8),
              (12, 11, 10), (14, 13, 12), (16, 15, 14)]:
        at = base.repeat(r)
        try:
            torch.cuda.reset_peak_memory_stats()
            d, L = data(at, rc, torch.float32, "cuda")
            t = bench(lambda: m(d, training=False, compute_stress=False), "cuda", 5, 2)
            print(f"  {len(at):>8}{L:>9}{t:>9.1f}{t*1000/len(at):>10.2f}"
                  f"{torch.cuda.max_memory_allocated()/1e6:>9.0f}M")
            del d
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  {len(at):>8}{L:>9}   OOM: {str(e)[:40]}")
            torch.cuda.empty_cache()
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(2)
    p = torch.cuda.get_device_properties(0)
    print(f"device : {torch.cuda.get_device_name(0)}  sm_{p.major}{p.minor}  "
          f"{p.multi_processor_count} SMs  {p.total_memory/1e9:.0f} GB")
    print(f"torch  : {torch.__version__}  (cuda {torch.version.cuda})")

    base = read("tests/cspbi3/results/relaxed/cubic_alpha_phase.vasp")
    _, rc = load(torch.float32, "cuda")
    m32, m64, at = correctness(base, rc)
    if not args.quick:
        dg, _ = data(at, rc, torch.float32, "cuda")
        gap1_batching(rc, base)
        gap2_compile(m32, dg)
        gap3_launches(m32, dg)
        gap4_syncs(m32, dg)
        gap5_determinism(rc, base)
        gap6_tf32(rc, base)
        scaling(rc, base)

    print("\n" + ("ALL CORRECTNESS CHECKS PASSED" if not FAILURES
                  else f"FAILURES: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
