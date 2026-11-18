#!/usr/bin/env python3
"""Self-driving architecture oracle search.

The candidate unit is a *new architecture*: a code source (binary, worktree,
commit or patch), a build command when needed, a flag template, and a small
inner knob sweep.  This tool does not invent code.  It decides which
architecture to measure, at which QPs, and when to stop.

Per architecture it reports three different bounds:

* an ideal/rate-free ceiling: source-segment replacement by default, or an
  explicitly supplied decoded-domain / ideal-future-reference replacement;
* the matched-rate K envelope over the inner knobs;
* the gap to the SVT 88.8 and AVM 90.2 walls.

Selection policy (deterministic):

1. Probe one or two arms at two QPs with segments.  Compute the ceiling.
2. If the ceiling cannot clear 88.8, close the gate.  Do not screen the rest,
   do not run a full ladder, do not sweep the knobs.
3. If it can clear, screen the remaining arms at two QPs (extending the QP set
   only when the target rate is not bracketed).
4. Promote the best screen plus the arms within a margin, capped at the
   finalist budget.  A pure-Python GP (UCB/EI) orders large grids; no heavy
   dependency is used.
5. Run the full QP ladder only for finalists.
6. If the full envelope is below 88.8, stop and report it.  Only an envelope
   that clears the wall unlocks the exhaustive sweep.

Scores are cached by (candidate, qp, md5) and complete ladders are cached by
their point hashes, so an interrupted or repeated run resumes without
re-encoding or re-scoring.

Usage:
  tools/oracle_search.py --bin build-release/av2enc
  tools/oracle_search.py --only nightly,eff-perc --bin /path/to/av2enc
  tools/oracle_search.py --bin /path/to/av2enc \
      --oracle /path/to/av2enc/tools/oracle.py

Outputs:
  docs/oracle-results.json          machine-readable architecture report
  docs/oracle-search.md             generated table in the marked section
  docs/oracle-decisions/<name>.md   per-candidate decision log
  .oracle-cache/                    score/ladder/encode cache (gitignored)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "docs" / "oracle-candidates.json"
DEFAULT_RESULTS = ROOT / "docs" / "oracle-results.json"
DEFAULT_DOC = ROOT / "docs" / "oracle-search.md"
DEFAULT_DECISIONS = ROOT / "docs" / "oracle-decisions"
DEFAULT_CACHE = ROOT / ".oracle-cache"
DEFAULT_BIN = ROOT / "build-release" / "av2enc"
DEFAULT_ORACLE = ROOT / "tools" / "oracle.py"

SVT_WALL = 88.8
AVM_WALL = 90.2
RESULTS_SCHEMA = "oracle-search/1"
CACHE_SCHEMA = "oracle-score-cache/1"

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


# --------------------------------------------------------------------------
# small utilities


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(text: str) -> str:
    return SLUG.sub("_", text).strip("_") or "candidate"


def fmt_value(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def render(text: str, values: dict, where: str) -> str:
    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise SystemExit(
                f"oracle_search: unknown placeholder {{{key}}} in {where}; "
                f"available: {', '.join(sorted(values))}")
        return fmt_value(values[key])

    return PLACEHOLDER.sub(replace, text)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def display_path(path: Path) -> str:
    # Keep the lexical path: testdata is a symlink to the shared fixture tree,
    # and the report should say `testdata/...`, not the symlink target.
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(ROOT.absolute()))
    except ValueError:
        return os.path.relpath(absolute, ROOT.absolute())


def tail(text: str | None, limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


# --------------------------------------------------------------------------
# registry


@dataclass
class Candidate:
    name: str
    status: str
    source: dict
    build: dict | None
    flags: list[str]
    env: dict[str, str]
    qp_ladder: list[int]
    knobs: dict[str, dict]
    ceiling: dict
    validation: dict
    notes: str
    raw: dict

    def arm_values(self) -> list[dict]:
        if not self.knobs:
            return [{}]
        names = list(self.knobs)
        value_lists = [self.knobs[name]["values"] for name in names]
        arms = []
        for combo in itertools.product(*value_lists):
            arms.append(dict(zip(names, combo, strict=True)))
        return arms

    def default_arm(self) -> dict:
        return {
            name: spec["default"]
            for name, spec in self.knobs.items()
        }

    def screen_qps(self) -> list[int]:
        ladder = self.qp_ladder
        if len(ladder) < 2:
            raise SystemExit(
                f"oracle_search: {self.name}: qp_ladder needs at least 2 QPs")
        middle = len(ladder) // 2
        return [ladder[middle - 1], ladder[middle]]

    def full_qps(self) -> list[int]:
        return list(self.qp_ladder)


@dataclass(frozen=True)
class Arm:
    values_json: str
    values: dict = field(compare=False, hash=False)

    @property
    def key(self) -> str:
        return self.values_json

    @property
    def digest(self) -> str:
        return sha256_text(self.values_json)[:12]

    def display(self) -> str:
        if not self.values:
            return "default"
        return ", ".join(f"{k}={fmt_value(v)}" for k, v in self.values.items())


def make_arms(cand: Candidate) -> list[Arm]:
    arms = []
    seen: set[str] = set()
    for values in cand.arm_values():
        key = canonical_json(values)
        if key in seen:
            raise SystemExit(
                f"oracle_search: {cand.name}: duplicate knob arm {key}")
        seen.add(key)
        arms.append(Arm(values_json=key, values=values))
    return arms


def parse_knobs(raw: dict, name: str) -> dict[str, dict]:
    knobs: dict[str, dict] = {}
    for knob, spec in raw.items():
        if isinstance(spec, list):
            values = list(spec)
            default = values[0] if values else None
            note = ""
        elif isinstance(spec, dict):
            values = list(spec.get("values", []))
            default = spec.get("default", values[0] if values else None)
            note = spec.get("note", "")
        else:
            raise SystemExit(
                f"oracle_search: {name}: knob {knob} must be a list or an object")
        if not values:
            raise SystemExit(
                f"oracle_search: {name}: knob {knob} has no values")
        if any(isinstance(value, (list, dict)) for value in values):
            raise SystemExit(
                f"oracle_search: {name}: knob {knob} values must be scalars")
        if default not in values:
            raise SystemExit(
                f"oracle_search: {name}: knob {knob} default {default!r} "
                "is not in its values")
        knobs[knob] = {
            "values": values,
            "default": default,
            "note": note,
        }
    return knobs


def load_registry(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"oracle_search: missing registry: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"oracle_search: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"oracle_search: {path}: registry must be an object")
    if raw.get("schema") != "oracle-candidates/1":
        raise SystemExit(
            f"oracle_search: {path}: expected schema oracle-candidates/1")
    if not isinstance(raw.get("candidates"), list):
        raise SystemExit(f"oracle_search: {path}: candidates must be a list")

    seen: set[str] = set()
    for entry in raw["candidates"]:
        if not isinstance(entry, dict):
            raise SystemExit("oracle_search: candidate entries must be objects")
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise SystemExit("oracle_search: every candidate needs a name")
        if name in seen:
            raise SystemExit(f"oracle_search: duplicate candidate {name}")
        seen.add(name)
        ladder = entry.get("qp_ladder")
        if (not isinstance(ladder, list) or len(ladder) < 2
                or any(not isinstance(qp, int) for qp in ladder)):
            raise SystemExit(
                f"oracle_search: {name}: qp_ladder must be at least two ints")
        if sorted(set(ladder)) != ladder:
            raise SystemExit(
                f"oracle_search: {name}: qp_ladder must be sorted and unique")
        flags = entry.get("flags")
        if isinstance(flags, str):
            pass
        elif isinstance(flags, list) and all(isinstance(x, str) for x in flags):
            pass
        else:
            raise SystemExit(
                f"oracle_search: {name}: flags must be a list or a string")
        env = entry.get("env", {})
        if not isinstance(env, dict) or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in env.items()):
            raise SystemExit(
                f"oracle_search: {name}: env must map strings to strings")
    return raw


def candidates_from_registry(raw: dict, only: set[str] | None,
                             statuses: set[str]) -> list[Candidate]:
    candidates = []
    for entry in raw["candidates"]:
        name = entry["name"]
        if only is not None and name not in only:
            continue
        status = entry.get("status", "open")
        if only is None and status not in statuses:
            continue
        flags = entry["flags"]
        if isinstance(flags, str):
            flags = shlex.split(flags)
        cand = Candidate(
            name=name,
            status=status,
            source=entry.get("source") or {"kind": "binary"},
            build=entry.get("build"),
            flags=list(flags),
            env=dict(entry.get("env", {})),
            qp_ladder=list(entry["qp_ladder"]),
            knobs=parse_knobs(entry.get("knobs", {}), name),
            ceiling=entry.get("ceiling") or {"kind": "source-segment"},
            validation=entry.get("validation", {}),
            notes=entry.get("notes", ""),
            raw=entry,
        )
        if not any("{qp}" in arg for arg in cand.flags):
            raise SystemExit(
                f"oracle_search: {name}: flag template must contain {{qp}}")
        if not any("{out}" in arg for arg in cand.flags):
            raise SystemExit(
                f"oracle_search: {name}: flag template must contain {{out}}")
        if len(make_arms(cand)) > 64:
            raise SystemExit(
                f"oracle_search: {name}: more than 64 inner arms; narrow the grid")
        candidates.append(cand)
    return candidates


# --------------------------------------------------------------------------
# cache


class Cache:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "scores.json"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "blobs").mkdir(exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)
        self.data = {
            "schema": CACHE_SCHEMA,
            "scores": {},
            "ladders": {},
            "encodes": {},
            "builds": {},
        }
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"oracle_search: corrupt cache {self.path}: {exc}") from exc
            if loaded.get("schema") != CACHE_SCHEMA:
                raise SystemExit(
                    f"oracle_search: cache {self.path} has schema "
                    f"{loaded.get('schema')!r}, expected {CACHE_SCHEMA}")
            for key in ("scores", "ladders", "encodes", "builds"):
                if isinstance(loaded.get(key), dict):
                    self.data[key] = loaded[key]
        self.hits = 0
        self.misses = 0

    def save(self) -> None:
        atomic_write(self.path, json.dumps(self.data, indent=1, sort_keys=True)
                     + "\n")

    def blob(self, md5: str) -> Path:
        return self.root / "blobs" / f"{md5}.ivf"

    def score_key(self, candidate: str, qp: int, md5: str) -> str:
        return f"{candidate}|q{qp}|{md5}"

    def get_score(self, candidate: str, qp: int, md5: str) -> dict | None:
        value = self.data["scores"].get(self.score_key(candidate, qp, md5))
        if value is not None:
            self.hits += 1
        return value

    def put_score(self, candidate: str, qp: int, md5: str, value: dict) -> None:
        self.data["scores"][self.score_key(candidate, qp, md5)] = value
        self.misses += 1

    def get_ladder(self, key: str) -> dict | None:
        value = self.data["ladders"].get(key)
        if value is not None:
            self.hits += 1
        return value

    def put_ladder(self, key: str, value: dict) -> None:
        self.data["ladders"][key] = value
        self.misses += 1

    def get_encode(self, key: str) -> dict | None:
        return self.data["encodes"].get(key)

    def put_encode(self, key: str, value: dict) -> None:
        self.data["encodes"][key] = value


# --------------------------------------------------------------------------
# oracle module and rate math


def load_oracle(path: Path):
    if not path.is_file():
        raise SystemExit(f"oracle_search: missing oracle: {path}")
    spec = importlib.util.spec_from_file_location("oracle_harness", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"oracle_search: cannot load oracle {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules; register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def match_rate_from_scores(oracle_mod, points: list[dict], target: float) -> dict:
    """Use oracle.py's own interpolation on cached point scores."""
    pseudo = [
        SimpleNamespace(
            label=point["label"],
            kbps=point["kbps"],
            metrics=SimpleNamespace(kpi=point["kpi"]),
        )
        for point in points
    ]
    matched = oracle_mod.match_rate(pseudo, target)
    matched["verdict"] = oracle_mod.verdict(matched["kpi"])
    return matched


# --------------------------------------------------------------------------
# GP surrogate (pure Python, no numpy)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j] - sum(
                lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 1e-12:
                    raise ValueError("matrix is not positive definite")
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def _solve_lower(lower: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    out = [0.0] * n
    for i in range(n):
        out[i] = (rhs[i] - sum(lower[i][j] * out[j]
                              for j in range(i))) / lower[i][i]
    return out


def _solve_upper_from_lower(lower: list[list[float]],
                            rhs: list[float]) -> list[float]:
    n = len(rhs)
    # Solve L^T x = rhs.
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (rhs[i] - sum(lower[j][i] * out[j]
                              for j in range(i + 1, n))) / lower[i][i]
    return out


def gp_fit_predict(x_train: list[list[float]], y_train: list[float],
                   x_test: list[list[float]]) -> tuple[list[float],
                                                       list[float]]:
    """GP with an RBF kernel on standardized inputs/outputs.

    Returns predictive mean and standard deviation in K units.  Raises
    ValueError when the problem is too small or the kernel is singular; the
    caller then falls back to measured screen order.
    """
    if len(x_train) < 2:
        raise ValueError("need at least two points")
    dims = len(x_train[0])
    x_mean = [sum(row[d] for row in x_train) / len(x_train)
              for d in range(dims)]
    x_std = []
    for d in range(dims):
        var = sum((row[d] - x_mean[d]) ** 2 for row in x_train) / len(x_train)
        x_std.append(math.sqrt(var) if var > 1e-12 else 1.0)
    y_mean = sum(y_train) / len(y_train)
    y_var = sum((value - y_mean) ** 2 for value in y_train) / len(y_train)
    y_std = math.sqrt(y_var) if y_var > 1e-12 else 1.0

    def standard(row: list[float]) -> list[float]:
        return [(row[d] - x_mean[d]) / x_std[d] for d in range(dims)]

    train = [standard(row) for row in x_train]
    test = [standard(row) for row in x_test]

    def kernel(a: list[float], b: list[float]) -> float:
        dist = sum((a[d] - b[d]) ** 2 for d in range(dims))
        return math.exp(-0.5 * dist)

    noise = 1e-3
    gram = [
        [kernel(train[i], train[j]) + (noise if i == j else 0.0)
         for j in range(len(train))]
        for i in range(len(train))
    ]
    lower = _cholesky(gram)
    centered = [(value - y_mean) / y_std for value in y_train]
    alpha = _solve_upper_from_lower(
        lower, _solve_lower(lower, centered))

    means = []
    sigmas = []
    for row in test:
        k = [kernel(row, point) for point in train]
        mean = sum(k[i] * alpha[i] for i in range(len(train)))
        v = _solve_lower(lower, k)
        variance = 1.0 - sum(v[i] * v[i] for i in range(len(v)))
        means.append(mean * y_std + y_mean)
        sigmas.append(math.sqrt(max(variance, 0.0)) * y_std)
    return means, sigmas


def gp_scores(arms: list[Arm], measured: dict[str, float],
              acquisition: str) -> dict[str, dict]:
    """Return GP prediction and acquisition value for every arm.

    The measured arms are the training set.  The acquisition is computed
    against the best measured K, so it is an improvement probability (EI) or
    an optimistic bound (UCB), not a second measurement.
    """
    result: dict[str, dict] = {}
    if len(measured) < 2:
        for arm in arms:
            result[arm.key] = {
                "mean": measured.get(arm.key),
                "sigma": None,
                "acquisition": None,
                "method": "fallback",
            }
        return result

    numeric_knobs = list(arms[0].values) if arms else []
    if any(isinstance(arm.values.get(k), str) for arm in arms
           for k in numeric_knobs):
        # One-hot is unnecessary for the small categorical grids used here:
        # use the registry index as a deterministic coordinate.
        index = {arm.key: i for i, arm in enumerate(arms)}
        x_train = [[float(index[key])] for key in measured]
        x_test = [[float(index[arm.key])] for arm in arms]
    else:
        x_train = [[float(arm.values.get(k, 0.0)) for k in numeric_knobs]
                   for arm in arms if arm.key in measured]
        x_test = [[float(arm.values.get(k, 0.0)) for k in numeric_knobs]
                  for arm in arms]
    y_train = [measured[arm.key] for arm in arms if arm.key in measured]
    try:
        means, sigmas = gp_fit_predict(x_train, y_train, x_test)
    except (ValueError, ZeroDivisionError):
        for arm in arms:
            result[arm.key] = {
                "mean": measured.get(arm.key),
                "sigma": None,
                "acquisition": None,
                "method": "fallback",
            }
        return result

    best = max(measured.values())

    def eager(values):
        return [value for value in values if value is not None]

    for arm, mean, sigma in zip(arms, means, sigmas, strict=True):
        sigma = max(sigma, 1e-9)
        z = (mean - best - 0.01) / sigma
        ei = (mean - best - 0.01) * _normal_cdf(z) + sigma * _normal_pdf(z)
        ucb = mean + 2.0 * sigma
        result[arm.key] = {
            "mean": round(mean, 6),
            "sigma": round(sigma, 6),
            "ei": round(ei, 6),
            "ucb": round(ucb, 6),
            "acquisition": round(ei if acquisition == "ei" else ucb, 6),
            "method": "gp-rbf",
        }
    return result


# --------------------------------------------------------------------------
# run context and encoder/build runner


@dataclass
class Context:
    args: argparse.Namespace
    registry: dict
    cache: Cache
    oracle_path: Path
    oracle: object
    ref: Path
    frames: int
    target: float
    threads: int
    bin_path: Path
    oracle_id: str
    ref_md5: str
    counters: dict = field(default_factory=lambda: {
        "encodes": 0,
        "encode_cache_hits": 0,
        "oracle_calls": 0,
        "oracle_cache_hits": 0,
        "score_cache_hits": 0,
    })
    workdirs: list[Path] = field(default_factory=list)

    def cleanup(self) -> None:
        if self.args.keep_work:
            return
        for path in self.workdirs:
            shutil.rmtree(path, ignore_errors=True)


def resolve_path(path_text: str, base: Path = ROOT) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.absolute()


def encode_key(cand: Candidate, arm: Arm, qp: int, binary: Path,
               ctx: Context) -> str:
    payload = {
        "candidate": cand.name,
        "arm": arm.values,
        "qp": qp,
        "flags": cand.flags,
        "env": cand.env,
        "binary_sha256": sha256_file(binary) if binary.is_file() else "missing",
        "ref_md5": ctx.ref_md5,
        "frames": ctx.frames,
        "threads": ctx.threads,
    }
    return sha256_text(canonical_json(payload))


def binary_for(cand: Candidate, ctx: Context) -> Path:
    source = cand.source
    kind = source.get("kind", "binary")
    if kind == "binary":
        if source.get("binary"):
            return resolve_path(source["binary"])
        return ctx.bin_path
    if kind == "worktree":
        worktree = source.get("worktree")
        binary = source.get("binary")
        if not worktree or not binary:
            raise SystemExit(
                f"oracle_search: {cand.name}: worktree sources need "
                "source.worktree and source.binary")
        return resolve_path(binary, resolve_path(worktree))
    if kind in ("commit", "branch", "patch"):
        return build_candidate(cand, ctx)
    raise SystemExit(
        f"oracle_search: {cand.name}: unknown source kind {kind!r}")


def build_candidate(cand: Candidate, ctx: Context) -> Path:
    """Build a commit/patch proposal in the cache, never in another worktree.

    A patch proposal is extracted with `git archive` rather than a linked
    worktree, so no other checkout is touched.
    """
    source = cand.source
    build = cand.build
    if not build or not build.get("command"):
        raise SystemExit(
            f"oracle_search: {cand.name}: commit/patch sources need a build "
            "command")
    commit = (source.get("commit") or source.get("branch")
              or source.get("base_commit"))
    if not commit:
        raise SystemExit(
            f"oracle_search: {cand.name}: commit/branch/patch source needs "
            "a commit, branch or base_commit")
    patch_path = None
    patch_md5 = None
    if source.get("kind") == "patch":
        patch_path = resolve_path(source.get("patch", ""))
        if not patch_path.is_file():
            raise SystemExit(
                f"oracle_search: {cand.name}: missing patch {patch_path}")
        patch_md5 = md5_file(patch_path)

    build_key = sha256_text(canonical_json({
        "candidate": cand.name,
        "commit": commit,
        "patch_md5": patch_md5,
        "build": build,
    }))[:16]
    build_root = ctx.cache.root / "builds" / build_key
    src_dir = build_root / "src"
    build_dir = build_root / "build"
    binary_text = build.get("binary", "{build}/av2enc")
    values = {
        "src": str(src_dir),
        "build": str(build_dir),
        "bin": str(build_root / "bin" / "av2enc"),
        "jobs": str(os.cpu_count() or 4),
        "root": str(ROOT),
    }
    binary = Path(render(binary_text, values, f"{cand.name} build.binary"))
    if not binary.is_absolute():
        binary = src_dir / binary
    if binary.exists():
        ctx.cache.data["builds"][build_key] = {
            "binary": str(binary),
            "commit": commit,
            "patch_md5": patch_md5,
        }
        return binary

    if not src_dir.is_dir():
        src_dir.mkdir(parents=True, exist_ok=True)
        archive = build_root / "src.tar"
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "archive", "--format=tar",
             "--output", str(archive), commit],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"oracle_search: {cand.name}: git archive {commit} failed:\n"
                f"{tail(proc.stderr)}")
        proc = subprocess.run(
            ["tar", "-xf", str(archive), "-C", str(src_dir)],
            capture_output=True, text=True)
        archive.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"oracle_search: {cand.name}: tar extraction failed:\n"
                f"{tail(proc.stderr)}")
        if patch_path is not None:
            proc = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch_path)],
                cwd=src_dir, capture_output=True, text=True)
            if proc.returncode != 0:
                raise SystemExit(
                    f"oracle_search: {cand.name}: patch apply failed:\n"
                    f"{tail(proc.stderr)}")

    command = build["command"]
    if isinstance(command, str):
        command = shlex.split(command)
    argv = [render(arg, values, f"{cand.name} build.command")
            for arg in command]
    build_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(argv, cwd=src_dir, capture_output=True, text=True)
    log_path = ctx.cache.root / "logs" / f"build-{build_key}.log"
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""))
    if proc.returncode != 0:
        raise SystemExit(
            f"oracle_search: {cand.name}: build failed (see {log_path}):\n"
            f"{tail(proc.stderr)}")
    if not binary.exists():
        raise SystemExit(
            f"oracle_search: {cand.name}: build finished but binary is "
            f"missing: {binary}")
    ctx.cache.data["builds"][build_key] = {
        "binary": str(binary),
        "commit": commit,
        "patch_md5": patch_md5,
    }
    return binary


@dataclass
class Artifact:
    qp: int
    path: Path
    md5: str
    bytes: int
    from_cache: bool


def encode_point(cand: Candidate, arm: Arm, qp: int, binary: Path,
                 ctx: Context) -> Artifact:
    key = encode_key(cand, arm, qp, binary, ctx)
    cached = ctx.cache.get_encode(key)
    if cached:
        blob = ctx.cache.blob(cached["md5"])
        if blob.is_file() and md5_file(blob) == cached["md5"]:
            ctx.counters["encode_cache_hits"] += 1
            return Artifact(qp=qp, path=blob, md5=cached["md5"],
                            bytes=cached["bytes"], from_cache=True)

    workdir = Path(tempfile.mkdtemp(prefix=f"oracle-{slug(cand.name)}-",
                                    dir=ctx.cache.root))
    ctx.workdirs.append(workdir)
    out_rel = f"out-{arm.digest}-q{qp}.ivf"
    values = {
        "bin": str(binary),
        "ref": str(ctx.ref),
        "out": out_rel,
        "frames": ctx.frames,
        "threads": ctx.threads,
        "qp": qp,
        **arm.values,
    }
    argv = [render(arg, values, f"{cand.name} flags") for arg in cand.flags]
    env = {k: v for k, v in os.environ.items() if not k.startswith("AV2ENC_")}
    env.update({
        name: render(value, values, f"{cand.name} env {name}")
        for name, value in cand.env.items()
    })
    proc = subprocess.run(argv, cwd=workdir, env=env,
                          capture_output=True, text=True,
                          timeout=ctx.args.encode_timeout or None)
    log_lines = [
        "$ " + " ".join(shlex.quote(arg) for arg in argv),
        f"cwd: {workdir}",
        f"returncode: {proc.returncode}",
        "",
        proc.stdout or "",
        proc.stderr or "",
    ]
    log_path = ctx.cache.root / "logs" / f"encode-{key[:16]}.log"
    log_path.write_text("\n".join(log_lines))
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cand.name} qp{qp} encode failed (see {log_path}):\n"
            f"{tail(proc.stderr)}")
    out_path = workdir / out_rel
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"{cand.name} qp{qp}: encoder did not write {out_path}")
    md5 = md5_file(out_path)
    size = out_path.stat().st_size
    blob = ctx.cache.blob(md5)
    if not blob.is_file():
        shutil.copyfile(out_path, blob)
    ctx.cache.put_encode(key, {
        "md5": md5,
        "bytes": size,
        "candidate": cand.name,
        "arm": arm.values,
        "qp": qp,
    })
    ctx.cache.save()
    ctx.counters["encodes"] += 1
    return Artifact(qp=qp, path=blob, md5=md5, bytes=size, from_cache=False)


# --------------------------------------------------------------------------
# oracle invocation


@dataclass
class Request:
    cand: Candidate
    arm: Arm
    qps: list[int]
    artifacts: list[Artifact]

    @property
    def group(self) -> str:
        return f"{slug(self.cand.name)}-{self.arm.digest[:8]}"


@dataclass
class EvalResult:
    cand: Candidate
    arm: Arm
    qps: list[int]
    point_md5: dict[int, str]
    points: list[dict]
    matched: dict
    segments: list[dict]
    segment_base: str | None
    from_ladder_cache: bool
    oracle_call: bool
    with_segments: bool

    def point_k(self) -> dict[int, float]:
        found = {}
        for point in self.points:
            label = point.get("label", "")
            match = re.search(r"@q(-?\d+)$", label)
            if match:
                found[int(match.group(1))] = point["kpi"]
            elif "qp" in point.get("point", ""):
                try:
                    found[int(point["point"].lstrip("q"))] = point["kpi"]
                except ValueError:
                    pass
        return found

    def as_cache(self) -> dict:
        return {
            "qps": self.qps,
            "point_md5": {str(qp): md5 for qp, md5 in self.point_md5.items()},
            "points": self.points,
            "matched": self.matched,
            "segments": self.segments,
            "segment_base": self.segment_base,
            "with_segments": self.with_segments,
        }


def result_from_cache(cand: Candidate, arm: Arm, value: dict,
                      with_segments: bool) -> EvalResult:
    return EvalResult(
        cand=cand,
        arm=arm,
        qps=[int(qp) for qp in value["qps"]],
        point_md5={int(qp): md5 for qp, md5 in value["point_md5"].items()},
        points=value["points"],
        matched=value["matched"],
        segments=value.get("segments", []),
        segment_base=value.get("segment_base"),
        from_ladder_cache=True,
        oracle_call=False,
        with_segments=with_segments,
    )


def ladder_key(cand: Candidate, arm: Arm, qps: list[int],
               artifacts: list[Artifact], ctx: Context,
               with_segments: bool) -> str:
    replacements = []
    for repl in ctx.args.replacement:
        if "=" not in repl:
            raise SystemExit(
                f"oracle_search: --replacement expects LABEL=Y4M, got {repl!r}")
        label, raw = repl.split("=", 1)
        path = resolve_path(raw)
        replacements.append({
            "label": label,
            "md5": md5_file(path) if path.is_file() else "missing",
        })
    payload = {
        "candidate": cand.name,
        "arm": arm.values,
        "points": [[qp, art.md5] for qp, art in zip(qps, artifacts,
                                                   strict=True)],
        "target_kbps": ctx.target,
        "oracle_sha256": ctx.oracle_id,
        "with_segments": with_segments,
        "replacements": replacements,
    }
    return sha256_text(canonical_json(payload))


def run_oracle(requests: list[Request], with_segments: bool,
               ctx: Context, tag: str) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix="oracle-call-", dir=ctx.cache.root))
    ctx.workdirs.append(workdir)
    report_path = workdir / "report.json"
    argv = [
        sys.executable,
        str(ctx.oracle_path),
        "--ref", str(ctx.ref),
        "--target-kbps", fmt_value(ctx.target),
        "--json", str(report_path),
    ]
    if ctx.args.decode_threads:
        argv += ["--decode-threads", str(ctx.args.decode_threads)]
    if not with_segments:
        argv.append("--no-segments")
    expected_qp: dict[str, int] = {}
    expected_group: dict[str, str] = {}
    for req in requests:
        for qp, art in zip(req.qps, req.artifacts, strict=True):
            label = f"{req.group}@q{qp}"
            argv += ["--ivf", f"{label}={art.path}"]
            expected_qp[label] = qp
            expected_group[label] = req.group
    for repl in ctx.args.replacement:
        if "=" not in repl:
            raise SystemExit(
                f"oracle_search: --replacement expects LABEL=Y4M, got {repl!r}")
        label, raw = repl.split("=", 1)
        path = resolve_path(raw)
        if not path.is_file():
            raise SystemExit(
                f"oracle_search: missing replacement {label}: {path}")
        argv += ["--decoded", f"{label}={path}"]
        argv += ["--segment-from", label]

    proc = subprocess.run(argv, capture_output=True, text=True,
                          timeout=ctx.args.oracle_timeout or None)
    log_path = ctx.cache.root / "logs" / f"oracle-{slug(tag)}.log"
    log_path.write_text(
        "$ " + " ".join(shlex.quote(arg) for arg in argv) + "\n\n"
        + (proc.stdout or "") + "\n" + (proc.stderr or ""))
    ctx.counters["oracle_calls"] += 1
    if proc.returncode != 0:
        raise RuntimeError(
            f"oracle failed for {tag} (see {log_path}):\n{tail(proc.stderr)}")
    if not report_path.is_file():
        raise RuntimeError(f"oracle wrote no JSON for {tag}")
    report = json.loads(report_path.read_text())
    by_group: dict[str, dict] = {}
    for entry in report.get("candidates", []):
        group = entry.get("candidate")
        if group not in {req.group for req in requests}:
            continue
        points = []
        for point in entry.get("points", []):
            label = point.get("label")
            qp = expected_qp.get(label)
            if qp is None:
                raise RuntimeError(
                    f"oracle returned unexpected point label {label!r}")
            points.append({
                "label": label,
                "qp": qp,
                "point": point.get("point"),
                "ivf_bytes": point.get("ivf_bytes"),
                "kbps": point.get("kbps"),
                "kpi": point.get("kpi"),
                "psnr": point.get("psnr"),
                "ssim": point.get("ssim"),
                "cvvdp_jod": point.get("cvvdp_jod"),
                "butteraugli": point.get("butteraugli"),
                "ssimulacra2": point.get("ssimulacra2"),
                "kpi_components": point.get("kpi_components"),
            })
        points.sort(key=lambda row: row["qp"])
        by_group[group] = {
            "matched": entry.get("matched"),
            "points": points,
            "segments": entry.get("segments", []),
            "segment_base": entry.get("segment_base"),
        }
    missing = [req.group for req in requests if req.group not in by_group]
    if missing:
        raise RuntimeError(
            f"oracle report omitted group(s): {', '.join(missing)}")
    return by_group


def evaluate_ladder(cand: Candidate, arm: Arm, qps: list[int],
                    ctx: Context, with_segments: bool,
                    purpose: str) -> EvalResult:
    binary = binary_for(cand, ctx)
    artifacts = [encode_point(cand, arm, qp, binary, ctx) for qp in qps]
    key = ladder_key(cand, arm, qps, artifacts, ctx, with_segments)
    cached = ctx.cache.get_ladder(key)
    if cached is not None:
        ctx.counters["oracle_cache_hits"] += 1
        return result_from_cache(cand, arm, cached, with_segments)

    if not with_segments:
        cached_points = []
        all_cached = True
        for qp, art in zip(qps, artifacts, strict=True):
            score = ctx.cache.get_score(cand.name, qp, art.md5)
            if score is None:
                all_cached = False
                break
            cached_points.append(score)
        if all_cached:
            ctx.counters["score_cache_hits"] += 1
            points = []
            for qp, art, score in zip(qps, artifacts, cached_points,
                                      strict=True):
                points.append({
                    "label": f"{slug(cand.name)}-{arm.digest[:8]}@q{qp}",
                    "qp": qp,
                    "point": f"q{qp}",
                    "ivf_bytes": score.get("ivf_bytes"),
                    "kbps": score["kbps"],
                    "kpi": score["kpi"],
                    "psnr": score.get("psnr"),
                    "ssim": score.get("ssim"),
                    "cvvdp_jod": score.get("cvvdp_jod"),
                    "butteraugli": score.get("butteraugli"),
                    "ssimulacra2": score.get("ssimulacra2"),
                    "kpi_components": score.get("kpi_components"),
                })
            matched = match_rate_from_scores(ctx.oracle, points, ctx.target)
            result = EvalResult(
                cand=cand, arm=arm, qps=qps,
                point_md5={qp: art.md5 for qp, art in zip(qps, artifacts,
                                                          strict=True)},
                points=points, matched=matched, segments=[],
                segment_base=None, from_ladder_cache=True,
                oracle_call=False, with_segments=False)
            ctx.cache.put_ladder(key, result.as_cache())
            ctx.cache.save()
            return result

    request = Request(cand=cand, arm=arm, qps=qps, artifacts=artifacts)
    report = run_oracle([request], with_segments, ctx,
                        f"{cand.name}-{arm.digest[:8]}-{purpose}")
    entry = report[request.group]
    points = entry["points"]
    for point, art in zip(points, artifacts, strict=True):
        ctx.cache.put_score(cand.name, point["qp"], art.md5, point)
    matched = entry["matched"]
    result = EvalResult(
        cand=cand, arm=arm, qps=qps,
        point_md5={qp: art.md5 for qp, art in zip(qps, artifacts,
                                                  strict=True)},
        points=points, matched=matched, segments=entry["segments"],
        segment_base=entry.get("segment_base"),
        from_ladder_cache=False, oracle_call=True,
        with_segments=with_segments)
    ctx.cache.put_ladder(key, result.as_cache())
    ctx.cache.save()
    return result


def choose_extension(cand: Candidate, used: list[int], matched: dict) -> int | None:
    ladder = cand.qp_ladder
    unused = [qp for qp in ladder if qp not in used]
    if not unused:
        return None
    kbps = matched.get("kbps")
    target = matched.get("target_kbps")
    if kbps is None or target is None:
        return unused[0]
    if kbps > target:
        higher = [qp for qp in unused if qp > max(used)]
        return higher[0] if higher else min(unused)
    lower = [qp for qp in unused if qp < min(used)]
    return lower[-1] if lower else max(unused)


def evaluate_bracketed(cand: Candidate, arm: Arm, qps: list[int],
                       ctx: Context, with_segments: bool,
                       purpose: str) -> EvalResult:
    used = sorted(set(qps))
    result = evaluate_ladder(cand, arm, used, ctx, with_segments, purpose)
    while (result.matched.get("out_of_range")
           or result.matched.get("method") == "nearest"):
        if len(used) >= min(len(cand.qp_ladder),
                            ctx.args.max_screen_qps):
            break
        extra = choose_extension(cand, used, result.matched)
        if extra is None:
            break
        used = sorted(used + [extra])
        result = evaluate_ladder(cand, arm, used, ctx, with_segments,
                                 purpose + "-extend")
    return result


# --------------------------------------------------------------------------
# ceiling, decisions and selection


def source_ceiling(cand: Candidate, evaluations: list[EvalResult],
                   ctx: Context) -> dict:
    rows = []
    declared = None
    kind = cand.ceiling.get("kind")
    if kind == "declared":
        declared = float(cand.ceiling["k"])
    elif kind not in (None, "source-segment", "auto", "decoded"):
        raise SystemExit(
            f"oracle_search: {cand.name}: unknown ceiling kind {kind!r}")

    invalid = [evaluation for evaluation in evaluations
               if evaluation.matched.get("out_of_range")]
    evaluations = [evaluation for evaluation in evaluations
                   if not evaluation.matched.get("out_of_range")]
    for evaluation in evaluations:
        base_k = evaluation.matched.get("kpi")
        if base_k is None:
            continue
        for segment in evaluation.segments:
            delta = segment.get("delta_k")
            if delta is None:
                continue
            replacement = segment.get("replacement") or "source"
            rows.append({
                "kind": replacement,
                "segment": segment.get("segment"),
                "delta_k": round(float(delta), 4),
                "k": round(float(base_k) + float(delta), 4),
                "base_k": round(float(base_k), 4),
                "base_kbps": evaluation.matched.get("kbps"),
                "arm": evaluation.arm.display(),
                "qps": evaluation.qps,
            })
    best = max(rows, key=lambda row: (row["k"], row["segment"])) if rows else None
    if declared is not None and (best is None or declared > best["k"]):
        best = {
            "kind": "declared",
            "segment": "declared envelope",
            "delta_k": None,
            "k": round(declared, 4),
            "base_k": None,
            "base_kbps": None,
            "arm": "registry",
            "qps": [],
        }
    return {
        "k": best["k"] if best else None,
        "method": best["kind"] if best else "unknown",
        "segment": best["segment"] if best else None,
        "delta_k": best["delta_k"] if best else None,
        "base_k": best["base_k"] if best else None,
        "rows": rows,
        "declared_k": declared,
        "invalid_rate": bool(invalid),
        "reason": "rate not bracketed" if invalid and not best else "",
    }


def decision_entry(step: str, arm: Arm, result: EvalResult | None,
                   decision: str, reason: str, **extra) -> dict:
    entry = {
        "step": step,
        "arm": arm.display(),
        "arm_key": arm.values,
        "decision": decision,
        "reason": reason,
    }
    if result is not None:
        entry.update({
            "qps": result.qps,
            "matched_k": round(float(result.matched["kpi"]), 4),
            "matched_kbps": round(float(result.matched["kbps"]), 4),
            "method": result.matched.get("method"),
            "out_of_range": bool(result.matched.get("out_of_range")),
            "from_ladder_cache": result.from_ladder_cache,
            "oracle_call": result.oracle_call,
            "segments": len(result.segments),
        })
    entry.update(extra)
    return entry


def gp_log(cand: Candidate, arms: list[Arm], measured: dict[str, float],
           acquisition: str) -> dict:
    scores = gp_scores(arms, measured, acquisition)
    return {
        "method": next(iter(scores.values()))["method"] if scores else "none",
        "acquisition": acquisition,
        "arms": {
            arm.display(): scores[arm.key] for arm in arms
        },
    }


def candidate_screen_qps(cand: Candidate, count: int) -> list[int]:
    ladder = cand.qp_ladder
    count = max(1, min(count, len(ladder)))
    start = (len(ladder) - count) // 2
    return ladder[start:start + count]


def run_candidate(cand: Candidate, ctx: Context) -> dict:
    arms = make_arms(cand)
    if not arms:
        raise SystemExit(f"oracle_search: {cand.name}: no arms")
    by_key = {arm.key: arm for arm in arms}
    default = Arm(canonical_json(cand.default_arm()), cand.default_arm())
    if default.key not in by_key:
        raise SystemExit(
            f"oracle_search: {cand.name}: default arm is not in the grid")
    decisions: list[dict] = []
    screens: dict[str, EvalResult] = {}
    full: dict[str, EvalResult] = {}
    ceilings: list[dict] = []

    def record(step, arm, result, decision, reason, **extra):
        decisions.append(decision_entry(step, arm, result, decision, reason,
                                        **extra))

    # Gate probes: the default arm plus the far end of the grid.  If neither
    # ceiling can clear the wall, the architecture is closed before any full
    # ladder or exhaustive sweep.
    probe_arms = [default]
    if len(arms) > 1 and ctx.args.gate_probes > 1:
        probe_arms.append(arms[-1])
    probe_arms = probe_arms[:max(1, ctx.args.gate_probes)]
    gate_qps = candidate_screen_qps(cand, ctx.args.screen_qps)
    for arm in probe_arms:
        result = evaluate_bracketed(cand, arm, gate_qps, ctx,
                                    with_segments=True, purpose="gate-probe")
        screens[arm.key] = result
        ceiling = source_ceiling(cand, [result], ctx)
        ceilings.append(ceiling)
        valid = not result.matched.get("out_of_range")
        extended = len(result.qps) > len(gate_qps)
        record("gate-probe", arm, result,
               "measure-ceiling" if valid else "invalid-probe",
               ("probe with source replacement"
                + (" (QP set extended to bracket 100 kbps)" if extended else ""))
               if valid else "probe did not bracket the target rate",
               ceiling_k=ceiling["k"], ceiling_method=ceiling["method"],
               ceiling_segment=ceiling["segment"])

    gate_k = max((row["k"] for row in ceilings
                  if row["k"] is not None), default=None)
    gate = {
        "state": "open",
        "ceiling_k": gate_k,
        "wall_k": ctx.args.svt_wall,
        "max_ceiling_k": max((row["k"] for row in ceilings
                              if row["k"] is not None), default=None),
        "margin_k": round(gate_k - ctx.args.svt_wall, 4)
        if gate_k is not None else None,
        "probes": [dict(row) for row in ceilings],
    }
    if gate_k is None:
        gate["state"] = "closed"
        gate["reason"] = "no ceiling could be measured"
    elif gate_k < ctx.args.svt_wall:
        gate["state"] = "closed"
        gate["reason"] = (
            f"ceiling {gate_k:.3f} < SVT {ctx.args.svt_wall:.1f}")
    else:
        gate["reason"] = (
            f"ceiling {gate_k:.3f} >= SVT {ctx.args.svt_wall:.1f}")

    if gate["state"] == "closed":
        record("gate", default, screens.get(default.key), "close",
               gate["reason"], ceiling_k=gate_k)
        result = finish_candidate(
            cand, arms, screens, full, gate, ceilings, decisions, ctx,
            stopped="gate")
        result["sweep"] = {
            "state": "skipped",
            "reason": "gate closed below the SVT wall; no full sweep",
        }
        result["validation_record"] = validation_record(cand, result)
        return result

    # Screen the rest of the grid.  A large grid is ordered by the GP
    # acquisition; the small grids here are screened exhaustively.
    unscreened = [arm for arm in arms if arm.key not in screens]
    while unscreened and len(screens) < ctx.args.max_screen_arms:
        if len(screens) >= 2 and len(unscreened) > 1:
            measured = {
                arm.key: float(screens[arm.key].matched["kpi"])
                for arm in arms if arm.key in screens
                and not screens[arm.key].matched.get("out_of_range")
            }
            scores = gp_scores(arms, measured, ctx.args.acquisition)
            ranked = sorted(
                unscreened,
                key=lambda arm: (
                    -(scores.get(arm.key, {}).get("acquisition") or -1e9),
                    arm.key,
                ))
            arm = ranked[0]
        else:
            arm = unscreened[0]
        result = evaluate_bracketed(cand, arm, gate_qps, ctx,
                                    with_segments=False, purpose="screen")
        screens[arm.key] = result
        ceiling = source_ceiling(cand, [result], ctx)
        if ceiling["k"] is not None:
            ceilings.append(ceiling)
        extended = len(result.qps) > len(gate_qps)
        record("screen", arm, result,
               "screen",
               ("screen without segments"
                + (" (QP set extended to bracket 100 kbps)" if extended else ""))
               if not result.matched.get("out_of_range")
               else "screen did not bracket the target rate",
               ceiling_k=ceiling["k"])
        unscreened = [row for row in arms if row.key not in screens]

    measured = {
        arm.key: float(screens[arm.key].matched["kpi"])
        for arm in arms if arm.key in screens
        and not screens[arm.key].matched.get("out_of_range")
    }
    if not measured:
        record("gate", default, screens.get(default.key), "close",
               "no screen arm bracketed the target rate")
        result = finish_candidate(
            cand, arms, screens, full, gate, ceilings, decisions, ctx,
            stopped="unbracketed")
        result["sweep"] = {
            "state": "skipped",
            "reason": "no screen arm bracketed the target rate",
        }
        result["validation_record"] = validation_record(cand, result)
        return result

    surrogate = gp_log(cand, arms, measured, ctx.args.acquisition)
    screen_best_key = max(measured, key=lambda key: (measured[key], key))
    threshold = measured[screen_best_key] - ctx.args.promotion_margin
    within = [arm for arm in arms
              if arm.key in measured and measured[arm.key] >= threshold]
    within.sort(key=lambda arm: (-measured[arm.key], arm.key))
    finalists = within[:max(1, ctx.args.max_finalists)]
    record("promote", by_key[screen_best_key], screens[screen_best_key],
           "promote",
           f"screen best {measured[screen_best_key]:.3f}; margin "
           f"{ctx.args.promotion_margin:.2f} keeps {len(within)}; "
           f"finalist budget {len(finalists)}",
           screened=len(measured), within=[arm.display() for arm in within],
           finalists=[arm.display() for arm in finalists])

    if ctx.args.full_ladders:
        for arm in finalists:
            result = evaluate_ladder(cand, arm, cand.full_qps(), ctx,
                                     with_segments=True, purpose="full")
            full[arm.key] = result
            ceiling = source_ceiling(cand, [result], ctx)
            ceilings.append(ceiling)
            valid = not result.matched.get("out_of_range")
            record("full-ladder", arm, result,
                   "measure-full" if valid else "invalid-full",
                   "full QP ladder with source segments",
                   ceiling_k=ceiling["k"], full_qps=result.qps)
    else:
        record("full-ladder", default, screens.get(default.key), "skipped",
               "--no-full-ladders")

    full_valid = {
        key: value for key, value in full.items()
        if not value.matched.get("out_of_range")
    }
    full_envelope = None
    if full_valid:
        best_full_key = max(
            full_valid,
            key=lambda key: (full_valid[key].matched["kpi"], key))
        full_envelope = float(full_valid[best_full_key].matched["kpi"])

    # The exhaustive sweep is unlocked only by a confirmed envelope above
    # the wall.  A candidate below the wall never buys the rest of the grid.
    sweep = None
    if (ctx.args.expand_above_wall and full_envelope is not None
            and full_envelope >= ctx.args.svt_wall):
        sweep = {"state": "run", "reason": "envelope clears SVT"}
        for arm in arms:
            if arm.key in full:
                continue
            if arm.key not in screens:
                result = evaluate_bracketed(cand, arm, gate_qps, ctx,
                                            with_segments=False,
                                            purpose="above-wall-screen")
                screens[arm.key] = result
            result = evaluate_ladder(cand, arm, cand.full_qps(), ctx,
                                     with_segments=True,
                                     purpose="above-wall-full")
            full[arm.key] = result
            record("above-wall-full-ladder", arm, result, "measure-full",
                   "only unlocked because the candidate cleared the wall",
                   full_qps=result.qps)
    elif full_envelope is not None and full_envelope < ctx.args.svt_wall:
        sweep = {
            "state": "skipped",
            "reason": f"full envelope {full_envelope:.3f} < "
                      f"SVT {ctx.args.svt_wall:.1f}; no full sweep",
        }
    else:
        sweep = {
            "state": "skipped",
            "reason": "no valid full ladder; screen is not an oracle claim",
        }

    full_valid = {
        key: value for key, value in full.items()
        if not value.matched.get("out_of_range")
    }
    if full_valid:
        best_full_key = max(
            full_valid,
            key=lambda key: (full_valid[key].matched["kpi"], key))
        best_k = float(full_valid[best_full_key].matched["kpi"])
        best_arm = full_valid[best_full_key].arm
        if best_k < ctx.args.svt_wall:
            record("decide", best_arm, full_valid[best_full_key], "no-go",
                   f"full envelope {best_k:.3f} < "
                   f"SVT {ctx.args.svt_wall:.1f}; stop, no full sweep")
        elif best_k < ctx.args.avm_wall:
            record("decide", best_arm, full_valid[best_full_key],
                   "clears-svt",
                   f"full envelope {best_k:.3f} clears SVT "
                   f"{ctx.args.svt_wall:.1f}; headroom to AVM "
                   f"{best_k - ctx.args.avm_wall:+.3f}")
        else:
            record("decide", best_arm, full_valid[best_full_key],
                   "clears-avm", "full envelope clears both walls")
    else:
        best_screen = max(
            (row for row in screens.values()
             if not row.matched.get("out_of_range")),
            key=lambda row: (row.matched["kpi"], row.arm.key))
        record("decide", best_screen.arm, best_screen, "screen-only",
               "no valid full ladder; do not promote a screen into a claim")

    result = finish_candidate(
        cand, arms, screens, full, gate, ceilings, decisions, ctx,
        stopped=None)
    result["surrogate"] = surrogate
    result["sweep"] = sweep
    result["validation_record"] = validation_record(cand, result)
    return result


def verdict_for(k: float, gate_state: str, svt_wall: float,
                avm_wall: float) -> str:
    if gate_state == "closed":
        return "gate closed below SVT 88.8"
    if k >= avm_wall:
        return "clears AVM 90.2"
    if k >= svt_wall:
        return "clears SVT 88.8"
    return "NO-GO: below SVT 88.8"


def finish_candidate(cand: Candidate, arms: list[Arm],
                     screens: dict[str, EvalResult],
                     full: dict[str, EvalResult], gate: dict,
                     ceilings: list[dict], decisions: list[dict],
                     ctx: Context, stopped: str | None) -> dict:
    valid_screens = {
        key: value for key, value in screens.items()
        if not value.matched.get("out_of_range")
    }
    best_screen = None
    if valid_screens:
        best_screen = max(valid_screens.values(),
                          key=lambda row: (row.matched["kpi"], row.arm.key))
    best_full = None
    valid_full = {
        key: value for key, value in full.items()
        if not value.matched.get("out_of_range")
    }
    if valid_full:
        best_full = max(valid_full.values(),
                        key=lambda row: (row.matched["kpi"], row.arm.key))
    envelope_k = None
    envelope_arm = None
    source = None
    if best_full is not None:
        envelope_k = round(float(best_full.matched["kpi"]), 4)
        envelope_arm = best_full.arm.display()
        source = "full"
    elif best_screen is not None:
        envelope_k = round(float(best_screen.matched["kpi"]), 4)
        envelope_arm = best_screen.arm.display()
        source = "screen"
    ceiling_k = max((row["k"] for row in ceilings
                     if row["k"] is not None), default=None)
    result = {
        "name": cand.name,
        "status": cand.status,
        "notes": cand.notes,
        "source": cand.source,
        "build": cand.build,
        "env": cand.env,
        "flags": cand.flags,
        "qp_ladder": cand.qp_ladder,
        "knobs": cand.knobs,
        "gate": gate,
        "screen": {
            "arms": [
                {
                    "arm": value.arm.display(),
                    "arm_key": value.arm.values,
                    "qps": value.qps,
                    "k": round(float(value.matched["kpi"]), 4),
                    "kbps": round(float(value.matched["kbps"]), 4),
                    "method": value.matched.get("method"),
                    "out_of_range": bool(value.matched.get("out_of_range")),
                    "ceiling_k": source_ceiling(cand, [value], ctx)["k"]
                    if value.with_segments else None,
                    "from_cache": value.from_ladder_cache,
                }
                for value in screens.values()
            ],
            "envelope_k": envelope_k if source == "screen" else (
                round(float(best_screen.matched["kpi"]), 4)
                if best_screen else None),
            "envelope_arm": best_screen.arm.display() if best_screen else None,
            "envelope_qps": best_screen.qps if best_screen else None,
        },
        "full": {
            "arms": [
                {
                    "arm": value.arm.display(),
                    "arm_key": value.arm.values,
                    "qps": value.qps,
                    "k": round(float(value.matched["kpi"]), 4),
                    "kbps": round(float(value.matched["kbps"]), 4),
                    "method": value.matched.get("method"),
                    "out_of_range": bool(value.matched.get("out_of_range")),
                    "from_cache": value.from_ladder_cache,
                }
                for value in full.values()
            ],
            "envelope_k": envelope_k if source == "full" else None,
            "envelope_arm": envelope_arm if source == "full" else None,
        },
        "envelope_k": envelope_k,
        "envelope_arm": envelope_arm,
        "envelope_arm_key": best_full.arm.values if best_full is not None
        else (best_screen.arm.values if best_screen is not None else None),
        "envelope_source": source,
        "ceiling_k": ceiling_k,
        "gap_svt": round(envelope_k - ctx.args.svt_wall, 4)
        if envelope_k is not None else None,
        "gap_avm": round(envelope_k - ctx.args.avm_wall, 4)
        if envelope_k is not None else None,
        "ceiling_gap_svt": round(ceiling_k - ctx.args.svt_wall, 4)
        if ceiling_k is not None else None,
        "verdict": verdict_for(envelope_k if envelope_k is not None else -1e9,
                               gate["state"], ctx.args.svt_wall,
                               ctx.args.avm_wall),
        "stopped": stopped,
        "decisions": decisions,
        "compute": {
            "arms": len(arms),
            "screens": len(screens),
            "full_ladders": len(full),
            "brute_force_full_ladders": len(arms),
            "saved_full_ladders": max(0, len(arms) - len(full)),
        },
    }
    return result


def validation_record(cand: Candidate, result: dict) -> dict:
    spec = cand.validation
    record: dict = {
        "candidate": cand.name,
        "note": spec.get("note", ""),
    }
    observed = result.get("envelope_k")
    record["observed_k"] = observed
    record["gate_state"] = result["gate"]["state"]
    record["full_ladders_run"] = result["compute"]["full_ladders"]
    record["brute_force_full_ladders"] = (
        spec.get("brute_force_full_ladders",
                 result["compute"]["brute_force_full_ladders"]))
    record["saved_full_ladders"] = (
        record["brute_force_full_ladders"]
        - record["full_ladders_run"])
    checks = []
    if "expect_gate" in spec:
        passed = result["gate"]["state"] == spec["expect_gate"]
        checks.append(passed)
        record["expected_gate"] = spec["expect_gate"]
        record["gate_check"] = "pass" if passed else "FAIL"
    if "known_best" in spec:
        target = spec["known_best"]
        selected = result.get("envelope_arm_key") or {}
        got = {key: selected.get(key) for key in target}
        found = (result.get("envelope_source") == "full"
                 and canonical_json(got) == canonical_json(target))
        checks.append(found)
        record["known_best"] = target
        record["selected_best"] = selected
        record["found_known_best"] = found
    if "reference_k" in spec:
        reference = float(spec["reference_k"])
        delta = (round(observed - reference, 4)
                 if observed is not None else None)
        reproduced = delta is not None and abs(delta) <= 0.5
        checks.append(reproduced)
        record["reference_k"] = reference
        record["reference"] = spec.get("reference", "")
        record["reference_delta"] = delta
        record["reproduced"] = reproduced
    if checks:
        record["passed"] = all(checks)
    else:
        record["passed"] = None
    return record


# --------------------------------------------------------------------------
# output


def write_decision_log(result: dict, out_dir: Path) -> Path:
    lines = [
        f"# {result['name']} — oracle decision log",
        "",
        f"- status: **{result['status']}**",
        f"- source: `{canonical_json(result['source'])}`",
        f"- gate: **{result['gate']['state']}** — {result['gate'].get('reason', '')}",
        f"- gate ceiling: **{fmt_optional(result['gate']['ceiling_k'])} K** "
        f"({result['gate'].get('reason', '')})",
        f"- measured max ceiling: **{fmt_optional(result['ceiling_k'])} K** "
        f"(SVT gap {fmt_optional(result['ceiling_gap_svt'])})",
        f"- envelope: **{fmt_optional(result['envelope_k'])} K** "
        f"from {result['envelope_source'] or 'none'} "
        f"({result['envelope_arm'] or '—'})",
        f"- full ladders: {result['compute']['full_ladders']} of "
        f"{result['compute']['brute_force_full_ladders']} arms "
        f"(saved {result['compute']['saved_full_ladders']})",
        f"- verdict: **{result['verdict']}**",
        "",
    ]
    if result.get("notes"):
        lines += [result["notes"], ""]
    if result.get("sweep"):
        lines += [
            f"Sweep: {result['sweep']['state']} — "
            f"{result['sweep'].get('reason', '')}",
            "",
        ]
    lines += [
        "| step | arm | qps | K | kbps | decision | reason |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for decision in result["decisions"]:
        qps = decision.get("qps")
        qps_text = ",".join(str(qp) for qp in qps) if qps else "—"
        k = decision.get("matched_k")
        kbps = decision.get("matched_kbps")
        lines.append(
            f"| {decision['step']} | {decision['arm']} | {qps_text} | "
            f"{fmt_optional(k)} | {fmt_optional(kbps)} | "
            f"{decision['decision']} | {decision['reason']} |")
    lines += [""]
    if result.get("surrogate"):
        surrogate = result["surrogate"]
        lines += [
            f"Surrogate: `{surrogate['method']}` with "
            f"{surrogate['acquisition'].upper()} acquisition.",
            "",
            "| arm | mean | sigma | EI | UCB |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, score in surrogate["arms"].items():
            lines.append(
                f"| {arm} | {fmt_optional(score.get('mean'))} | "
                f"{fmt_optional(score.get('sigma'))} | "
                f"{fmt_optional(score.get('ei'))} | "
                f"{fmt_optional(score.get('ucb'))} |")
        lines += [""]
    path = out_dir / f"{slug(result['name'])}.md"
    atomic_write(path, "\n".join(lines) + "\n")
    return path


def fmt_optional(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(results: list[dict], ctx: Context) -> str:
    lines = [
        "<!-- oracle-search:generated:begin -->",
        "## 10. Generated architecture search",
        "",
        f"Fixture: `{display_path(ctx.ref)}`, {ctx.frames} frames, "
        f"target {ctx.target:.3f} kbps. Walls: SVT "
        f"{ctx.args.svt_wall:.1f}, AVM {ctx.args.avm_wall:.1f}. "
        f"Oracle: `{display_path(ctx.oracle_path)}`.",
        "",
        "| candidate | gate | gate ceiling | screen envelope | full envelope | "
        "gap SVT | gap AVM | full ladders | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        screen = result["screen"].get("envelope_k")
        full = result["full"].get("envelope_k")
        lines.append(
            f"| **{result['name']}** | {result['gate']['state']} | "
            f"{fmt_optional(result['gate'].get('ceiling_k'))} | "
            f"{fmt_optional(screen)} | {fmt_optional(full)} | "
            f"{fmt_optional(result['gap_svt'])} | "
            f"{fmt_optional(result['gap_avm'])} | "
            f"{result['compute']['full_ladders']}/"
            f"{result['compute']['brute_force_full_ladders']} | "
            f"{result['verdict']} |")
    lines += [""]

    any_svt = any((result["envelope_k"] or -1e9) >= ctx.args.svt_wall
                  for result in results)
    any_avm = any((result["envelope_k"] or -1e9) >= ctx.args.avm_wall
                  for result in results)
    if not any_svt:
        lines += ["**No candidate clears 88.8.**", ""]
    elif not any_avm:
        lines += ["A candidate clears 88.8 but none clears 90.2.", ""]
    else:
        lines += ["A candidate clears 90.2.", ""]
    lines += [
        "A ceiling above the wall is not a claim: only the matched-rate K "
        "from a bracketing ladder is. The gate uses the ceiling to decide "
        "whether an architecture is worth a full ladder; it never promotes a "
        "ceiling row into the result column.",
        "",
    ]

    validation = [result["validation_record"] for result in results
                  if result.get("validation_record", {}).get("passed")
                  is not None or "reference_k" in
                  result.get("validation_record", {})]
    if validation:
        lines += [
            "### Validation",
            "",
            "| candidate | check | observed | result |",
            "|---|---|---|---|",
        ]
        for record in validation:
            if "known_best" in record:
                lines.append(
                    f"| {record['candidate']} | known best "
                    f"`{canonical_json(record['known_best'])}` | "
                    f"`{canonical_json(record.get('selected_best'))}` | "
                    f"{'pass' if record.get('found_known_best') else 'FAIL'} |")
            if "expected_gate" in record:
                lines.append(
                    f"| {record['candidate']} | gate expected "
                    f"{record['expected_gate']} | "
                    f"{record['gate_state']} | {record['gate_check']} |")
            if "reference_k" in record:
                lines.append(
                    f"| {record['candidate']} | reference K "
                    f"{record['reference_k']:.2f} | "
                    f"{fmt_optional(record.get('observed_k'))} | "
                    f"{'pass' if record.get('reproduced') else 'FAIL'} |")
        lines.append("")
        for record in validation:
            lines.append(
                f"- {record['candidate']}: {record['full_ladders_run']} full "
                f"ladder(s) vs {record['brute_force_full_ladders']} for brute "
                f"force; saved {record['saved_full_ladders']}.")
        lines.append("")

    lines += [
        "### Decision logs",
        "",
        "Per-candidate logs: " + ", ".join(
            f"[`{slug(result['name'])}.md`](oracle-decisions/{slug(result['name'])}.md)"
            for result in results) + ".",
        "",
        "<!-- oracle-search:generated:end -->",
    ]
    return "\n".join(lines) + "\n"


def update_doc(path: Path, generated: str) -> None:
    begin = "<!-- oracle-search:generated:begin -->"
    end = "<!-- oracle-search:generated:end -->"
    if path.is_file():
        text = path.read_text()
    else:
        text = "# Oracle search\n"
    if begin in text and end in text:
        before, rest = text.split(begin, 1)
        _, after = rest.split(end, 1)
        updated = before.rstrip() + "\n\n" + generated + after.lstrip("\n")
    else:
        updated = text.rstrip() + "\n\n" + generated
    atomic_write(path, updated)


def write_results(path: Path, results: list[dict], ctx: Context,
                  validation: dict) -> dict:
    summary = {
        "candidates_run": len(results),
        "best_envelope_k": max(
            (row["envelope_k"] for row in results
             if row["envelope_k"] is not None), default=None),
        "best_candidate": max(
            (row for row in results if row["envelope_k"] is not None),
            key=lambda row: (row["envelope_k"], row["name"]),
            default={}).get("name"),
        "any_candidate_clears_svt": any(
            (row["envelope_k"] or -1e9) >= ctx.args.svt_wall
            for row in results),
        "any_candidate_clears_avm": any(
            (row["envelope_k"] or -1e9) >= ctx.args.avm_wall
            for row in results),
        "any_ceiling_clears_svt": any(
            (row["ceiling_k"] or -1e9) >= ctx.args.svt_wall
            for row in results),
        "no_candidate_clears_88_8": not any(
            (row["envelope_k"] or -1e9) >= ctx.args.svt_wall
            for row in results),
        "statement": (
            "No candidate clears 88.8."
            if not any((row["envelope_k"] or -1e9) >= ctx.args.svt_wall
                       for row in results)
            else "At least one candidate clears 88.8."),
    }
    payload = {
        "schema": RESULTS_SCHEMA,
        "fixture": {
            "ref": display_path(ctx.ref),
            "frames": ctx.frames,
            "target_kbps": ctx.target,
            "threads": ctx.threads,
        },
        "walls": {
            "svt": ctx.args.svt_wall,
            "avm": ctx.args.avm_wall,
        },
        "policy": {
            "screen_qps": ctx.args.screen_qps,
            "max_screen_qps": ctx.args.max_screen_qps,
            "max_screen_arms": ctx.args.max_screen_arms,
            "max_finalists": ctx.args.max_finalists,
            "promotion_margin_k": ctx.args.promotion_margin,
            "gate_probes": ctx.args.gate_probes,
            "acquisition": ctx.args.acquisition,
            "expand_above_wall": ctx.args.expand_above_wall,
            "segments": ctx.args.segments,
        },
        "oracle": {
            "path": display_path(ctx.oracle_path),
            "sha256": ctx.oracle_id,
        },
        "bin": display_path(ctx.bin_path),
        "candidates": results,
        "summary": summary,
        "validation": validation,
        "cache": {
            "path": display_path(ctx.cache.path),
            "hits": ctx.cache.hits,
            "misses": ctx.cache.misses,
            "encodes": ctx.counters["encodes"],
            "encode_cache_hits": ctx.counters["encode_cache_hits"],
            "oracle_calls": ctx.counters["oracle_calls"],
            "oracle_cache_hits": ctx.counters["oracle_cache_hits"],
            "score_cache_hits": ctx.counters["score_cache_hits"],
        },
    }
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return payload


# --------------------------------------------------------------------------
# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Self-driving architecture oracle search.  Candidate is "
                    "code + flags; knobs are its inner sweep.")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--bin", default=str(DEFAULT_BIN),
                    help="default encoder binary (default: build-release/av2enc)")
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE,
                    help="tools/oracle.py to invoke (default: this tree's)")
    ap.add_argument("--ref", type=Path, default=None,
                    help="override the fixture reference")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--target-kbps", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated candidate names")
    ap.add_argument("--status", default="open",
                    help="comma-separated statuses to run when --only is "
                         "absent (default: open)")
    ap.add_argument("--screen-qps", type=int, default=None)
    ap.add_argument("--max-screen-qps", type=int, default=None)
    ap.add_argument("--max-screen-arms", type=int, default=None)
    ap.add_argument("--max-finalists", type=int, default=None)
    ap.add_argument("--promotion-margin", type=float, default=None)
    ap.add_argument("--gate-probes", type=int, default=None)
    ap.add_argument("--acquisition", choices=("ei", "ucb"), default=None)
    ap.add_argument("--svt-wall", type=float, default=SVT_WALL)
    ap.add_argument("--avm-wall", type=float, default=AVM_WALL)
    ap.add_argument("--no-full-ladders", dest="full_ladders",
                    action="store_false", default=True)
    ap.add_argument("--no-expand-above-wall", dest="expand_above_wall",
                    action="store_false", default=None)
    ap.add_argument("--no-segments", dest="segments", action="store_false",
                    default=None,
                    help="debug only: skips the ceiling; the gate then closes")
    ap.add_argument("--replacement", action="append", default=[],
                    metavar="LABEL=Y4M",
                    help="decoded-domain replacement for an ideal future "
                         "reference / ceiling; repeatable")
    ap.add_argument("--decode-threads", default=None)
    ap.add_argument("--encode-timeout", type=float, default=0)
    ap.add_argument("--oracle-timeout", type=float, default=0)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved candidates and arms, run nothing")
    return ap.parse_args(argv)


def apply_defaults(args: argparse.Namespace, registry: dict) -> None:
    defaults = registry.get("defaults", {})
    if args.screen_qps is None:
        args.screen_qps = int(defaults.get("screen_qps", 2))
    if args.max_screen_qps is None:
        args.max_screen_qps = int(defaults.get("max_screen_qps", 4))
    if args.max_screen_arms is None:
        args.max_screen_arms = int(defaults.get("max_screen_arms", 8))
    if args.max_finalists is None:
        args.max_finalists = int(defaults.get("max_finalists", 2))
    if args.promotion_margin is None:
        args.promotion_margin = float(
            defaults.get("promotion_margin_k", 0.5))
    if args.gate_probes is None:
        args.gate_probes = int(defaults.get("gate_probes", 2))
    if args.acquisition is None:
        args.acquisition = defaults.get("acquisition", "ei")
    if args.expand_above_wall is None:
        args.expand_above_wall = bool(
            defaults.get("expand_above_wall", True))
    if args.segments is None:
        args.segments = bool(defaults.get("segments", True))
    fixture = registry.get("fixture", {})
    walls = registry.get("walls", {})
    if args.svt_wall == SVT_WALL and "svt" in walls:
        args.svt_wall = float(walls["svt"])
    if args.avm_wall == AVM_WALL and "avm" in walls:
        args.avm_wall = float(walls["avm"])
    if args.ref is None:
        args.ref = resolve_path(fixture.get("ref", "testdata/nasa_480x240_30fps_60f.y4m"))
    if args.frames is None:
        args.frames = int(fixture.get("frames", 60))
    if args.target_kbps is None:
        args.target_kbps = float(fixture.get("target_kbps", 100.0))
    if args.threads is None:
        args.threads = int(fixture.get("threads", 8))
    if args.screen_qps < 2:
        raise SystemExit("oracle_search: --screen-qps must be at least 2")
    if args.max_screen_qps < args.screen_qps:
        raise SystemExit(
            "oracle_search: --max-screen-qps must be >= --screen-qps")
    if args.max_finalists < 1:
        raise SystemExit("oracle_search: --max-finalists must be positive")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = load_registry(args.registry)
    apply_defaults(args, registry)
    only = ({name.strip() for name in args.only.split(",") if name.strip()}
            if args.only else None)
    statuses = {name.strip() for name in args.status.split(",")
                if name.strip()}
    candidates = candidates_from_registry(registry, only, statuses)
    if not candidates:
        raise SystemExit("oracle_search: no candidates selected")

    args.ref = args.ref.expanduser().absolute()
    args.bin = str(resolve_path(args.bin))
    args.oracle = args.oracle.expanduser().absolute()
    args.cache_dir = args.cache_dir.expanduser().absolute()
    if not args.ref.is_file():
        raise SystemExit(f"oracle_search: missing reference: {args.ref}")
    oracle = load_oracle(args.oracle)
    oracle_id = sha256_file(args.oracle)
    ctx = Context(
        args=args,
        registry=registry,
        cache=Cache(args.cache_dir),
        oracle_path=args.oracle,
        oracle=oracle,
        ref=args.ref,
        frames=args.frames,
        target=args.target_kbps,
        threads=args.threads,
        bin_path=Path(args.bin),
        oracle_id=oracle_id,
        ref_md5=md5_file(args.ref),
    )

    if args.dry_run:
        for cand in candidates:
            arms = make_arms(cand)
            print(f"{cand.name}: {len(arms)} arm(s), "
                  f"gate probes {min(len(arms), max(1, args.gate_probes))}, "
                  f"screen qps {cand.screen_qps()}, full {cand.full_qps()}")
            for arm in arms:
                print(f"  - {arm.display()}")
        ctx.cleanup()
        return 0

    if not ctx.bin_path.is_file():
        # Only binary/worktree candidates need this default; a commit/patch
        # candidate builds its own.  Fail early with a useful message.
        if any((cand.source.get("kind", "binary") == "binary")
               for cand in candidates):
            raise SystemExit(
                f"oracle_search: missing --bin {ctx.bin_path}; pass "
                "--bin /path/to/av2enc or set source.binary in the registry")

    results = []
    errors = []
    try:
        for cand in candidates:
            print(f"oracle_search: {cand.name}", file=sys.stderr)
            try:
                result = run_candidate(cand, ctx)
                results.append(result)
                log_path = write_decision_log(result, args.decisions)
                print(f"  gate={result['gate']['state']} "
                      f"ceiling={fmt_optional(result['ceiling_k'])} "
                      f"envelope={fmt_optional(result['envelope_k'])} "
                      f"log={display_path(log_path)}", file=sys.stderr)
            except (RuntimeError, SystemExit, OSError,
                    subprocess.TimeoutExpired) as exc:
                message = str(exc)
                errors.append({"candidate": cand.name, "error": message})
                print(f"  FAILED: {message}", file=sys.stderr)
        if not results and errors:
            raise SystemExit(
                "oracle_search: every candidate failed:\n" +
                "\n".join(f"- {row['candidate']}: {row['error']}"
                          for row in errors))
        validation_records = [row.get("validation_record")
                              for row in results
                              if row.get("validation_record")]
        validation_checks = [record["passed"] for record in validation_records
                             if record.get("passed") is not None]
        validation = {
            "records": validation_records,
            "passed": all(validation_checks) if validation_checks else None,
            "note": "Validation is a harness check on a known grid, not a "
                    "claim that the grid clears the wall.",
            "errors": errors,
        }
        payload = write_results(args.results, results, ctx, validation)
        update_doc(args.doc, markdown_table(results, ctx))
        print(f"oracle_search: wrote {display_path(args.results)}")
        print(payload["summary"]["statement"])
        if errors:
            print("oracle_search: candidate errors:", file=sys.stderr)
            for row in errors:
                print(f"  - {row['candidate']}: {row['error']}",
                      file=sys.stderr)
        return 0
    finally:
        ctx.cache.save()
        ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
