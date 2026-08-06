"""dragrace command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .case import Case
from .config import Config


def _case_paths(root: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(root.rglob("*.yaml")):
        if p.stem.startswith("sweep_"):
            continue
        out[p.stem] = p
    return out


def _config_paths(root: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(root.glob("*.yaml"))}


def _load_all(repo: Path):
    from . import adapter as adapter_mod
    adapter_mod.discover(repo / "adapters")
    cases = {k: Case.from_yaml(v) for k, v in _case_paths(repo / "cases").items()}
    configs = {k: Config.from_yaml(v) for k, v in _config_paths(repo / "configs").items()}
    return cases, configs, adapter_mod


# --------------------------------------------------------------- commands --
def cmd_list(args, repo: Path) -> int:
    cases, configs, adapter_mod = _load_all(repo)
    print("CASES")
    for c in cases.values():
        print(f"  {c.summary()}")
    print("\nCONFIGS")
    for c in configs.values():
        print(f"  {c.summary()}   env={c.conda_env}")
    print("\nADAPTERS")
    for name in adapter_mod.available():
        a = adapter_mod.get(name)
        print(f"  {name:16s} status={a.status}")
    return 0


def cmd_doctor(args, repo: Path) -> int:
    from . import backend, fingerprint
    cases, configs, adapter_mod = _load_all(repo)

    m = fingerprint.machine()
    print(f"machine   {m['cpu']}  ({m['cpu_vendor']}, {m['logical_cores']} logical cores)")
    print(f"          {m['platform']}  python {m['python']}")
    print(f"          id={m['id']}")
    if m["gpus"]:
        for g in m["gpus"]:
            print(f"gpu       {g['name']} driver {g['driver']} {g['memory']}")
    else:
        print("gpu       none detected -- gpu_* configs will be skipped")

    print(f"\nblas      {backend.detect_blas()}")
    for e in backend.detect_thread_counts():
        print(f"          {e['api']:<10} threads={e['threads']:<4} {e['path']}")
    print(f"fft       importable: {backend.available_fft_backends()}")
    print(f"numpy.fft.fft2 -> {backend.numpy_fft_module()}")

    if m["cpu_vendor"] == "amd":
        print("\n  NOTE: MKL dispatches conservatively on AMD parts and the "
              "MKL_DEBUG_CPU_TYPE\n        workaround was removed in MKL 2020. Treat "
              "cpu_mkl_* results here as\n        measuring MKL-on-AMD, not the "
              "propagators. See envs/README.md.")

    print("\nSUPPORT MATRIX  (adapter x config, for each case's algorithm class)")
    for cname, case in cases.items():
        print(f"\n  case {cname}  [{case.algorithm_class}]")
        for aname in adapter_mod.available():
            ad = adapter_mod.get(aname)
            cells = []
            for cfgname, cfg in configs.items():
                sup = ad.check_requirements() and ad.supports(case, cfg)
                cells.append(f"{cfgname}={'yes' if sup else 'no'}")
            print(f"    {aname:16s} " + "  ".join(cells))
            for cfgname, cfg in configs.items():
                sup = ad.check_requirements() and ad.supports(case, cfg)
                if not sup:
                    print(f"      {cfgname}: {sup.reason}")
                    break
    return 0


def cmd_run(args, repo: Path) -> int:
    from . import fingerprint
    from .runner import RunSpec, new_run_id, run_one

    cases, configs, _ = _load_all(repo)
    cpaths, gpaths = _case_paths(repo / "cases"), _config_paths(repo / "configs")
    if args.case not in cases:
        print(f"unknown case {args.case!r}; known: {sorted(cases)}", file=sys.stderr)
        return 2
    if args.config not in configs:
        print(f"unknown config {args.config!r}; known: {sorted(configs)}", file=sys.stderr)
        return 2

    spec = RunSpec(cases[args.case], configs[args.config], args.adapter, args.mode,
                   str(cpaths[args.case]), str(gpaths[args.config]))
    run_id = args.run_id or new_run_id()
    res = run_one(spec, repo / "results", run_id, fingerprint.machine()["id"],
                  allow_fallback=args.allow_env_fallback,
                  strict_backend=not args.no_strict_backend, repo_root=repo)
    _print_result(res)
    return 0 if res.get("status") in ("ok", "unsupported", "skipped") else 1


def cmd_sweep(args, repo: Path) -> int:
    from . import fingerprint
    from .runner import RunSpec, new_run_id, run_one

    cases, configs, adapter_mod = _load_all(repo)
    cpaths, gpaths = _case_paths(repo / "cases"), _config_paths(repo / "configs")

    sel_cases = args.cases or list(cases)
    sel_configs = args.configs or list(configs)
    sel_adapters = args.adapters or adapter_mod.available()
    run_id = args.run_id or new_run_id()
    machine_id = fingerprint.machine()["id"]

    print(f"run_id={run_id}  {len(sel_cases)}x{len(sel_configs)}x{len(sel_adapters)} "
          f"x {len(args.modes)} modes")
    tally: dict[str, int] = {}
    for cname in sel_cases:
        for gname in sel_configs:
            for aname in sel_adapters:
                for mode in args.modes:
                    spec = RunSpec(cases[cname], configs[gname], aname, mode,
                                   str(cpaths[cname]), str(gpaths[gname]))
                    res = run_one(spec, repo / "results", run_id, machine_id,
                                  allow_fallback=args.allow_env_fallback,
                                  strict_backend=not args.no_strict_backend,
                                  repo_root=repo)
                    st = res.get("status", "?")
                    tally[st] = tally.get(st, 0) + 1
                    flag = {"ok": "  ", "unsupported": "--", "skipped": "--"}.get(st, "!!")
                    print(f"  {flag} {aname:16s} {gname:16s} {cname:24s} {mode:9s} {st}")
                    if st not in ("ok", "unsupported", "skipped"):
                        print(f"       {str(res.get('reason', ''))[:160]}")
    print(f"\n{tally}")
    print(f"results in results/raw/*/{run_id}/")
    return 0


def cmd_ledger(args, repo: Path) -> int:
    """Print the kernel-shape ledger for one adapter+case, without a full run."""
    from .flops import ledger as ledger_mod, ideal_work

    cases, configs, adapter_mod = _load_all(repo)
    case, cfg = cases[args.case], configs[args.config]
    ad = adapter_mod.get(args.adapter)
    sup = ad.supports(case, cfg)
    if not sup:
        print(f"unsupported: {sup.reason}")
        return 1
    # configure/build run inside the patched context so that adapters which
    # cache a callable at configure time pick up the instrumented version;
    # their calls are then dropped so only the propagation is priced.
    with ledger_mod.record() as led:
        ad.configure(cfg)
        state = ad.build(case, cfg)
        ad.sync(ad.propagate(state))
        led.reset()
        ad.sync(ad.propagate(state))
    ideal = ideal_work(case)
    print(f"{args.adapter} @ {case.id} [{cfg.id}]\n")
    print(led.render())
    print(f"\nideal (model):  {ideal.flops / 1e9:.4f} GFLOP   [{ideal.detail}]")
    if led.flops:
        print(f"algorithmic overhead A = ledger/ideal = {led.flops / ideal.flops:.3f}")
    print("\nNote: `@` between plain ndarrays bypasses np.matmul and is invisible "
          "to the ledger,\nso MFT-heavy codes under-count GEMMs here. See "
          "src/dragrace/flops/ledger.py.")
    return 0


def cmd_machine(args, repo: Path) -> int:
    from .flops.roofline import measure_machine
    m = measure_machine(quick=args.quick)
    print(json.dumps(m.to_dict(), indent=2))
    out = repo / "results" / "machine.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m.to_dict(), indent=2))
    print(f"\nwritten to {out}")
    return 0


def cmd_report(args, repo: Path) -> int:
    from .report import aggregate, render_text
    rows = aggregate(repo / "results")
    if not rows:
        print("no results found; run `dragrace sweep` first")
        return 1
    print(render_text(rows))
    out = repo / "results" / "index.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n{len(rows)} results -> {out}")
    return 0


def _print_result(res: dict) -> None:
    st = res.get("status")
    print(f"status={st}  {res.get('adapter', {}).get('name')} "
          f"{res.get('case_id')} [{res.get('config_id')}] {res.get('mode')}")
    if "accuracy" in res:
        a = res["accuracy"]
        print(f"  accuracy: rel_l2={a.get('rel_l2'):.3e} gate={a.get('gate')} "
              f"scale={a.get('scale_abs'):.6g} conj={a.get('conjugated')}")
    if "gradient_accuracy" in res:
        g = res["gradient_accuracy"]
        print(f"  gradient: max_rel_err={g.get('max_rel_err'):.3e} "
              f"cos={g.get('cosine_similarity'):.12f} gate={g.get('gate')}")
    if "timing" in res and res["timing"].get("device_compute_stats"):
        s = res["timing"]["device_compute_stats"]
        print(f"  timing:   median={s['median'] * 1e3:.3f} ms  min={s['min'] * 1e3:.3f} ms")
    if "flops" in res and "ideal" in res["flops"]:
        print(f"  ideal:    {res['flops']['ideal']['flops'] / 1e9:.4f} GFLOP")
    if "memory" in res and res["memory"]:
        m = res["memory"]
        if m.get("tracemalloc_peak_bytes"):
            print(f"  memory:   tracemalloc peak={m['tracemalloc_peak_bytes'] / 2**20:.1f} MiB")
    if st not in ("ok", "unsupported", "skipped"):
        print(f"  reason:   {res.get('reason')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dragrace", description=__doc__)
    ap.add_argument("--repo", default=".", help="repository root")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list cases, configs and adapters")
    sub.add_parser("doctor", help="environment, backends and the support matrix")

    def add_common(p):
        p.add_argument("--allow-env-fallback", action="store_true",
                       help="run in the active interpreter when the config's conda env "
                            "is missing; results are not attributable")
        p.add_argument("--no-strict-backend", action="store_true",
                       help="permit resolved != requested backend (exploration only)")
        p.add_argument("--run-id", default=None)

    p = sub.add_parser("run", help="one adapter x case x config")
    p.add_argument("--case", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--config", default="cpu_numpy_1t")
    p.add_argument("--mode", default="timing",
                   choices=["timing", "memory", "ledger", "trace", "gradient", "all"])
    add_common(p)

    p = sub.add_parser("sweep", help="cross product of cases, configs and adapters")
    p.add_argument("--cases", nargs="*")
    p.add_argument("--configs", nargs="*")
    p.add_argument("--adapters", nargs="*")
    p.add_argument("--modes", nargs="*", default=["timing"])
    add_common(p)

    p = sub.add_parser("ledger", help="print the kernel-shape ledger for one run")
    p.add_argument("--case", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--config", default="cpu_numpy_1t")

    p = sub.add_parser("machine", help="measure roofline peaks (zgemm + STREAM)")
    p.add_argument("--quick", action="store_true")

    sub.add_parser("report", help="aggregate results")

    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    return {
        "list": cmd_list, "doctor": cmd_doctor, "run": cmd_run, "sweep": cmd_sweep,
        "ledger": cmd_ledger, "machine": cmd_machine, "report": cmd_report,
    }[args.cmd](args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
