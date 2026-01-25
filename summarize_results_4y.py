#!/usr/bin/env python3
"""
Summarize experiments under results_4y/ into a spreadsheet-friendly table.

This project logs epochs starting at 0, so a run with num_epochs=N is considered complete
when the last testing epoch equals N-1 (e.g. 99/100).

Default behavior:
- Only include experiments that look complete based on testing.log last epoch.
- Read config from either <exp_dir>/config.yaml or a single *.yaml in <exp_dir>.
- Read logs from <exp_dir>/training.log and <exp_dir>/testing.log (if present).
- Write CSV (always) and XLSX (when --out-xlsx is provided and openpyxl is available).

Usage:
  python3 summarize_results_4y.py --results-dir results_4y --out results_4y_summary.csv --out-xlsx results_4y_summary.xlsx
  python3 summarize_results_4y.py --include-incomplete
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _coerce_scalar(value: str) -> Any:
    v = value.strip()
    if v == "":
        return None
    vl = v.lower()
    if vl in {"true", "false"}:
        return vl == "true"

    # Strip quotes
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]

    # Int / float
    try:
        if re.fullmatch(r"[+-]?\d+", v):
            return int(v)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", v) or re.fullmatch(
            r"[+-]?\d+(?:[eE][+-]?\d+)", v
        ):
            return float(v)
    except Exception:
        pass

    return v


def parse_flat_yaml(path: Path) -> Dict[str, Any]:
    """
    Minimal YAML reader for this project configs (flat key: value pairs).
    Avoids adding dependencies (PyYAML) and is robust enough for current config format.
    """
    data: Dict[str, Any] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return data

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        data[key] = _coerce_scalar(val)
    return data


@dataclass
class TestLogSummary:
    best_acc: Optional[float] = None
    best_epoch: Optional[int] = None
    best_loss: Optional[float] = None
    last_loss: Optional[float] = None
    last_acc: Optional[float] = None
    last_epoch: Optional[int] = None
    last_total: Optional[int] = None
    max_total_epochs_seen: Optional[int] = None
    last_acc_per_class: Optional[Dict[str, float]] = None


TEST_LINE_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+)\s*/\s*(?P<total>\d+).*?Test Loss:\s*(?P<loss>[-+0-9.eE]+).*?Test Accuracy:\s*(?P<acc>[-+0-9.eE]+)",
    re.IGNORECASE,
)


PER_CLASS_RE = re.compile(
    r"Test Accuracy per class:\s*(?P<names>\[[^\]]*\])\s*=\s*(?P<vals>\[[^\]]*\])",
    re.IGNORECASE,
)


def _parse_py_list_literal(s: str) -> Optional[list]:
    try:
        v = ast.literal_eval(s)
    except Exception:
        return None
    return v if isinstance(v, list) else None


def _parse_class_names_list(s: str) -> Optional[List[str]]:
    """
    Parse class names from a python-ish list literal like:
      "['정상', '관심', '경고', '위험']"
    """
    names = _parse_py_list_literal(s)
    if names is None:
        return None
    out: List[str] = []
    for n in names:
        out.append(str(n))
    return out


def _parse_float_list_brackets(s: str) -> Optional[List[float]]:
    """
    Parse numeric list from strings like:
      "[87.5, 69.9, 47.2, 84.0]"  (commas)
      "[87.5 69.9 47.2 84.0]"    (whitespace, numpy-style)
    """
    ss = s.strip()
    if not (ss.startswith("[") and ss.endswith("]")):
        return None
    inner = ss[1:-1].strip()
    if inner == "":
        return []
    # normalize separators
    inner = inner.replace(",", " ")
    parts = [p for p in inner.split() if p]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            return None
    return out


def parse_testing_log(path: Path) -> TestLogSummary:
    out = TestLogSummary()
    if not path.exists():
        return out

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = TEST_LINE_RE.search(line)
                if not m:
                    continue
                epoch = int(m.group("epoch"))
                total = int(m.group("total"))
                acc = float(m.group("acc"))
                loss = float(m.group("loss"))

                out.max_total_epochs_seen = (
                    total
                    if out.max_total_epochs_seen is None
                    else max(out.max_total_epochs_seen, total)
                )

                # last
                if out.last_epoch is None or epoch >= out.last_epoch:
                    out.last_epoch = epoch
                    out.last_acc = acc
                    out.last_loss = loss
                    out.last_total = total
                    pm = PER_CLASS_RE.search(line)
                    if pm:
                        names = _parse_class_names_list(pm.group("names"))
                        vals = _parse_float_list_brackets(pm.group("vals"))
                        if names and vals and len(names) == len(vals):
                            per: Dict[str, float] = {}
                            for n, v in zip(names, vals):
                                try:
                                    per[str(n)] = float(v)
                                except Exception:
                                    continue
                            out.last_acc_per_class = per if per else None

                # best
                if out.best_acc is None or acc > out.best_acc:
                    out.best_acc = acc
                    out.best_epoch = epoch
                    out.best_loss = loss
    except Exception:
        # Keep best-effort behavior
        return out

    return out


def infer_total_epochs_from_log_line(line: str) -> Optional[int]:
    m = re.search(r"Epoch:\s*\d+\s*/\s*(\d+)", line)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


ARGS_LINE_RE = re.compile(r"Arguments:\s*Namespace\((.*)\)")


def _split_kv_pairs(s: str) -> List[str]:
    """
    Split 'k=v, k2=v2, ...' into ['k=v', 'k2=v2', ...] using a conservative rule:
    split on ', ' only when the next token looks like an identifier followed by '='.
    """
    s = s.strip()
    if not s:
        return []
    return re.split(r",\s(?=[A-Za-z_]\w*\s*=)", s)


def parse_arguments_pairs(args_str: str) -> List[Tuple[str, str]]:
    """
    Parse either:
      - "Namespace(k=v, ...)"  or
      - "k=v, k2=v2, ..." (fallback format)

    Returns list of (key, raw_value_str) preserving order.
    """
    s = (args_str or "").strip()
    if not s:
        return []

    if s.startswith("Namespace(") and s.endswith(")"):
        inner = s[len("Namespace(") : -1].strip()
    else:
        inner = s

    pairs: List[Tuple[str, str]] = []
    for part in _split_kv_pairs(inner):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        pairs.append((k, v))
    return pairs


def format_arguments_pairs(pairs: List[Tuple[str, str]]) -> str:
    if not pairs:
        return ""
    inner = ", ".join([f"{k}={v}" for k, v in pairs])
    return f"Namespace({inner})"


def extract_arguments_string(training_log: Path, cfg: Dict[str, Any]) -> str:
    """
    Prefer the CLI Namespace(...) printed into training.log, because it contains the full
    arguments exactly as run. Fallback to a compact config key=value list.
    """
    if training_log.exists():
        try:
            with training_log.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = ARGS_LINE_RE.search(line)
                    if m:
                        return f"Namespace({m.group(1)})"
        except Exception:
            pass

    # fallback: config as key=value pairs (sorted for stability)
    parts: List[str] = []
    for k in sorted(cfg.keys()):
        parts.append(f"{k}={cfg[k]}")
    return ", ".join(parts)


def find_config_file(exp_dir: Path) -> Optional[Path]:
    cfg = exp_dir / "config.yaml"
    if cfg.exists():
        return cfg

    yamls = [p for p in exp_dir.glob("*.yaml") if p.is_file()]
    if len(yamls) == 1:
        return yamls[0]
    if len(yamls) > 1:
        # Prefer one that looks like "config" if multiple exist
        for p in yamls:
            if p.name.lower() in {"config.yaml", "configs.yaml"}:
                return p
        # Otherwise choose the most recently modified
        yamls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return yamls[0]
    return None


def safe_get(d: Dict[str, Any], key: str) -> Any:
    return d.get(key, None)


def summarize_one(exp_dir: Path) -> Dict[str, Any]:
    config_path = find_config_file(exp_dir)
    cfg: Dict[str, Any] = parse_flat_yaml(config_path) if config_path else {}

    training_log = exp_dir / "training.log"
    testing_log = exp_dir / "testing.log"

    # Determine target epochs
    target_epochs = safe_get(cfg, "num_epochs")
    if not isinstance(target_epochs, int):
        # try to infer from logs
        inferred: Optional[int] = None
        if inferred is None and testing_log.exists():
            test_sum = parse_testing_log(testing_log)
            inferred = test_sum.max_total_epochs_seen
        target_epochs = inferred if inferred is not None else 0

    test_sum = parse_testing_log(testing_log)
    # Completion rule for this project:
    # epochs are logged starting at 0, so final epoch is num_epochs - 1.
    completed = False
    if target_epochs and test_sum.last_epoch is not None:
        completed = (test_sum.last_epoch == int(target_epochs) - 1) and (
            test_sum.last_total in (None, int(target_epochs))
        )

    row: Dict[str, Any] = {
        "run_dir": exp_dir.name,
        "completed": completed,
        "num_epochs": int(target_epochs) if target_epochs else "",
        "last_test_loss": test_sum.last_loss if test_sum.last_loss is not None else "",
        "last_test_acc": test_sum.last_acc if test_sum.last_acc is not None else "",
        "last_test_epoch": test_sum.last_epoch if test_sum.last_epoch is not None else "",
    }

    # last line per-class accuracy (if present)
    if test_sum.last_acc_per_class:
        for k, v in test_sum.last_acc_per_class.items():
            row[f"last_test_acc_class_{k}"] = v

    # Add common config fields (stable columns for sheets)
    for k in [
        "comment",
        "data",
        "modality",
        "seed",
        "batch_size",
        "lr",
        "scheduler",
        "step_size",
        "num_layers",
        "dim",
        "imb_ratio",
        "use_textemb",
        "use_dim_matching_layer",
        "use_paco",
        "use_cb",
        "use_mtm",
        "alpha",
        "beta",
        "gamma",
        "moco_k",
        "moco_m",
        "moco_t",
        "lamb_paco_fusion",
        "lamb_ce_fusion",
        "lamb_mtm_fusion",
        "mtm_lambda",
        "feat_dim",
        "num_classes",
    ]:
        row[k] = safe_get(cfg, k) if k in cfg else ""

    return row


def iter_experiment_dirs(results_dir: Path) -> Iterable[Path]:
    for entry in sorted(results_dir.iterdir()):
        if entry.is_dir():
            yield entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_4y", help="Directory containing experiment subfolders")
    ap.add_argument("--out", default="results_4y_summary.csv", help="Output CSV path")
    ap.add_argument(
        "--out-xlsx",
        default="",
        help="Optional output XLSX path (enables red highlighting for differing args)",
    )
    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Only include runs for this dataset (repeatable). Example: --dataset sms",
    )
    ap.add_argument(
        "--diff-against",
        choices=["previous", "first"],
        default="previous",
        help="How to compare argument columns for red highlighting in XLSX",
    )
    ap.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also include incomplete runs (completed=false)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"results-dir not found: {results_dir}")

    rows: List[Dict[str, Any]] = []
    for exp_dir in iter_experiment_dirs(results_dir):
        row = summarize_one(exp_dir)
        # dataset filter
        if args.dataset:
            wanted: set = set()
            for d in args.dataset:
                for part in str(d).split(","):
                    part = part.strip()
                    if part:
                        wanted.add(part)
            if wanted and row.get("data", "") not in wanted:
                continue
        if (not args.include_incomplete) and (not row["completed"]):
            continue
        rows.append(row)

    # Drop best_* columns from output (user requested)
    for r in rows:
        r.pop("best_test_acc", None)
        r.pop("best_test_epoch", None)

    # Ensure per-class accuracy columns exist for all rows (stable header)
    class_names: List[str] = []
    class_set: set = set()
    for r in rows:
        for k in list(r.keys()):
            if k.startswith("last_test_acc_class_"):
                name = k[len("last_test_acc_class_") :]
                if name not in class_set:
                    class_set.add(name)
                    class_names.append(name)
    for r in rows:
        for name in class_names:
            r.setdefault(f"last_test_acc_class_{name}", "")

    # Drop columns that have the same value across all rows (keep run_dir always)
    mandatory_cols = {"run_dir"}
    if rows:
        all_keys: List[str] = []
        seen: set = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)

        constant_cols: set = set()
        for k in all_keys:
            if k in mandatory_cols:
                continue
            vals = {str(r.get(k, "")) for r in rows}
            if len(vals) == 1:
                constant_cols.add(k)

        if constant_cols:
            for r in rows:
                for k in constant_cols:
                    r.pop(k, None)

    # Stable header order (make metrics first)
    class_cols = [f"last_test_acc_class_{name}" for name in class_names]
    preferred = [
        "data",
        "last_test_acc",
        *class_cols,
        "last_test_loss",
        "last_test_epoch",
        "completed",
        "num_epochs",
    ]

    header: List[str] = []
    for k in preferred:
        if any(k in r for r in rows) and k not in header:
            header.append(k)
    for r in rows:
        for k in r.keys():
            if k not in header:
                header.append(k)
    # User requested: run_dir should be the last column
    if any("run_dir" in r for r in rows):
        header = [k for k in header if k != "run_dir"] + ["run_dir"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")

    # Optional XLSX with red highlighting
    if args.out_xlsx:
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font
        except Exception as e:
            print(f"Skipping XLSX (openpyxl not available): {e}")
            return 0

        wb = Workbook()
        ws = wb.active
        ws.title = "results_4y"

        # write header
        ws.append(header)

        # determine which columns are "arguments" columns for highlighting
        arg_keys = [
            "seed",
            "batch_size",
            "lr",
            "scheduler",
            "step_size",
            "num_layers",
            "dim",
            "imb_ratio",
            "use_textemb",
            "use_dim_matching_layer",
            "use_paco",
            "use_cb",
            "use_mtm",
            "alpha",
            "beta",
            "gamma",
            "moco_k",
            "moco_m",
            "moco_t",
            "lamb_paco_fusion",
            "lamb_ce_fusion",
            "lamb_mtm_fusion",
            "mtm_lambda",
            "feat_dim",
            "num_classes",
        ]
        arg_col_idxs = [header.index(k) + 1 for k in arg_keys if k in header]  # 1-based for openpyxl

        red_font = Font(color="FF0000")

        # write rows and highlight diffs
        for i, r in enumerate(rows):
            ws.append([r.get(k, "") for k in header])

            # row index in sheet (1 header row + i data rows)
            row_idx = 2 + i
            if i == 0:
                continue

            if args.diff_against == "previous":
                base = rows[i - 1]
            else:
                base = rows[0]

            for col_idx in arg_col_idxs:
                key = header[col_idx - 1]
                cur_v = r.get(key, "")
                base_v = base.get(key, "")
                if cur_v != base_v:
                    ws.cell(row=row_idx, column=col_idx).font = red_font

        out_xlsx = Path(args.out_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        wb.save(out_xlsx)
        print(f"Wrote XLSX to {out_xlsx}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

