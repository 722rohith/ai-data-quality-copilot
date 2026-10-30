"""Data-quality engine: profiling, hidden-null detection, rules, schema + distribution drift,
ETL log parsing and a reviewable cleaning plan that yields Pandas code, SQL and a cleaned DataFrame."""
import json
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype, is_bool_dtype

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "?", "-", "--",
               "unknown", "missing", "#n/a"}
ID_TOKENS = {"id", "identifier", "code", "key", "year", "zip", "pin", "phone", "number"}
POSITIVE_PREFIX = ("weight", "price", "amount", "quant", "visib", "salary",
                   "cost", "revenue", "sales", "height", "mrp")
POSITIVE_EXACT = {"age", "qty", "amt"}

# A label that differs from another only by one of these is a DIFFERENT category
# (e.g. "Dollars" vs "Dollars (millions)"), never a spelling variant.
_SCALE = re.compile(r"million|billion|thousand|trillion|percent|\bpct\b|\bbn\b|\bmn\b|[%$€£]", re.I)


def _tokens(name):
    return [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]


def is_id_like(name, s=None):
    toks = _tokens(name)
    if any(t in ID_TOKENS for t in toks):
        return True
    if s is not None and is_numeric_dtype(s) and len(s) > 20:
        nn = s.dropna()
        if len(nn) and nn.nunique() == len(nn) and (nn % 1 == 0).all():
            return True
    return False


def expects_positive(name):
    toks = _tokens(name)
    return any(t in POSITIVE_EXACT or t.startswith(POSITIVE_PREFIX) for t in toks)


def _is_text(s):
    return (is_object_dtype(s) or is_string_dtype(s)) and not is_bool_dtype(s)


def _label_groups(values):
    """Find labels that are likely the same category written differently.

    Returns (groups, conflicts):
      groups    -> list of (set_of_labels, confidence) where confidence is
                   "high"   (only case / spacing / _ / - differ -> safe to auto-merge) or
                   "review" (prefix or initials match -> a human must confirm)
      conflicts -> pairs that look similar but differ by a scale/unit word
                   (e.g. 'Dollars' vs 'Dollars (millions)'); NEVER merged.
    """
    norm = {}
    for v in values:
        norm.setdefault(re.sub(r"[\s_\-]+", " ", v.strip().lower()), []).append(v)
    keys = list(norm)
    groups = [(set(g), "high") for g in norm.values() if len(g) > 1]
    conflicts = []
    for a in keys:
        for b in keys:
            if a == b or len(a) >= len(b):
                continue
            words = b.split()
            abbrev = len(words) > 1 and a == "".join(w[0] for w in words)
            prefix = len(a) >= 3 and b.startswith(a)
            if prefix:
                extra = b[len(a):]
                if _SCALE.search(extra):
                    conflicts.append((sorted(norm[a])[0], sorted(norm[b])[0]))
                    continue
                if re.search(r"\d", extra):      # 'Level 1' vs 'Level 10' etc.
                    continue
            if abbrev or prefix:
                groups.append((set(norm[a]) | set(norm[b]), "review"))
    merged = []
    for g, conf in groups:
        for m in merged:
            if m[0] & g:
                m[0].update(g)
                if conf == "review":
                    m[1] = "review"
                break
        else:
            merged.append([set(g), conf])
    return [(s, c) for s, c in merged], conflicts


# ----------------------------------------------------------------------
# PROFILING
# ----------------------------------------------------------------------
def profile_dataframe(df):
    n = len(df)
    cols = {}
    for c in df.columns:
        s = df[c]
        info = {"dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True)), "hidden_nulls": 0,
                "hidden_tokens": [], "invalid_zeros": 0, "outliers": 0,
                "numeric_text": False, "bad_numeric_text": 0,
                "label_groups": [], "label_review": [], "scale_conflicts": [],
                "nonstandard_rows": 0, "review_rows": 0, "id_like": False}
        info["missing_pct"] = round(info["missing"] / n * 100, 2) if n else 0.0

        if _is_text(s):
            st = s.dropna().astype(str)
            mask = st.str.strip().str.lower().isin(NULL_TOKENS)
            info["hidden_nulls"] = int(mask.sum())
            info["hidden_tokens"] = sorted(st[mask].unique().tolist())[:10]
            clean = st[~mask]
            num = pd.to_numeric(clean, errors="coerce")
            if len(clean) and num.notna().mean() >= 0.8:
                info["numeric_text"] = True
                info["bad_numeric_text"] = int(num.isna().sum())
            elif 1 < clean.nunique() <= 30:
                counts = clean.value_counts()
                groups, conflicts = _label_groups(counts.index.tolist())
                for members, conf in groups:
                    # Split by normalised spelling: case/spacing-only differences are always safe;
                    # only merging *across* different spellings needs a human decision.
                    sub = {}
                    for v in members:
                        sub.setdefault(re.sub(r"[\s_\-]+", " ", v.strip().lower()), []).append(v)
                    reps = []
                    for vs in sub.values():
                        rep = max(vs, key=lambda v: counts.get(v, 0))
                        reps.append(rep)
                        extra = sorted(set(vs) - {rep})
                        if extra:
                            info["label_groups"].append({"canonical": rep, "variants": extra})
                            info["nonstandard_rows"] += int(sum(counts.get(v, 0) for v in extra))
                    if len(sub) > 1:
                        total = {k: sum(counts.get(x, 0) for x in vs) for k, vs in sub.items()}
                        top_key = max(total, key=total.get)
                        canon = max(sub[top_key], key=lambda v: counts.get(v, 0))
                        # every spelling of the other sub-groups maps straight to the canonical label
                        variants = sorted(x for k, vs in sub.items() if k != top_key for x in vs)
                        info["label_review"].append({"canonical": canon, "variants": variants})
                        info["review_rows"] += int(sum(counts.get(x, 0) for x in variants))
                info["scale_conflicts"] = conflicts[:5]
        elif is_numeric_dtype(s) and not is_bool_dtype(s):
            info["id_like"] = is_id_like(c, s)
            nn = s.dropna()
            if not info["id_like"] and len(nn) > 10:
                if expects_positive(c):
                    info["invalid_zeros"] = int((nn == 0).sum() + (nn < 0).sum())
                q1, q3 = nn.quantile(.25), nn.quantile(.75)
                iqr = q3 - q1
                if iqr > 0:
                    info["outliers"] = int(((nn < q1 - 1.5 * iqr) | (nn > q3 + 1.5 * iqr)).sum())
        cols[c] = info

    prof = {"rows": n, "columns_count": len(df.columns),
            "duplicates": int(df.duplicated().sum()),
            "missing_cells": int(df.isna().sum().sum()), "columns": cols}
    prof["total_rows"] = n
    prof["duplicate_rows"] = prof["duplicates"]
    prof["missing_percentage"] = {c: i["missing_pct"] for c, i in cols.items()}
    prof["score"] = quality_score(prof)
    return prof


def hidden_total(prof):
    """Cells that look filled in but are placeholders, invalid, or high-confidence label variants."""
    return sum(i["hidden_nulls"] + i["invalid_zeros"] + i["nonstandard_rows"] + i["bad_numeric_text"]
               for i in prof["columns"].values())


def quality_score(prof):
    """Honest score: average gaps + duplicates + invalid values + a penalty for
    every column with more than 5% missing, so one bad column cannot hide.
    'Review' label groups are not penalised: they are not confirmed errors."""
    n = max(prof["rows"], 1)
    cols = prof["columns"].values()
    k = max(len(prof["columns"]), 1)
    avg_missing = sum(i["missing_pct"] for i in cols) / k
    dup = prof["duplicates"] / n * 100
    invalid = sum((i["hidden_nulls"] + i["invalid_zeros"] + i["nonstandard_rows"]
                   + i["bad_numeric_text"]) / n * 100 for i in cols) / k
    col_penalty = sum(min(10, (i["missing_pct"] - 5) / 2) for i in cols if i["missing_pct"] > 5)
    score = max(0.0, min(100.0, 100 - avg_missing - dup - invalid - col_penalty))
    label = ("Excellent" if score >= 90 else "Good" if score >= 75
             else "Fair" if score >= 60 else "Needs work")
    return {"value": round(score, 1), "label": label,
            "avg_missing": round(avg_missing, 2), "duplicates": round(dup, 2),
            "invalid": round(invalid, 2), "column_penalty": round(col_penalty, 2)}


# ----------------------------------------------------------------------
# RULES
# ----------------------------------------------------------------------
RULE_OPS = ["not_null", "unique", ">", ">=", "<", "<=", "between", "in_set", "regex"]


def evaluate_rule(df, rule):
    col, op, val = rule.get("column"), rule.get("op"), str(rule.get("value", "") or "").strip()
    text = f"{col} {op} {val}".strip()
    if col not in df.columns:
        return {"rule": text, "status": "ERROR", "failed": 0, "pct": 0.0, "note": "unknown column"}
    s = df[col]
    try:
        if op == "not_null":
            bad = s.isna()
        elif op == "unique":
            bad = s.notna() & s.duplicated(keep=False)
        elif op in (">", ">=", "<", "<="):
            x = pd.to_numeric(s, errors="coerce"); v = float(val)
            ok = {">": x > v, ">=": x >= v, "<": x < v, "<=": x <= v}[op]
            bad = x.notna() & ~ok
        elif op == "between":
            lo, hi = [float(t) for t in val.split(",")]
            x = pd.to_numeric(s, errors="coerce")
            bad = x.notna() & ~x.between(lo, hi)
        elif op == "in_set":
            allowed = {t.strip() for t in val.split(",")}
            bad = s.notna() & ~s.astype(str).isin(allowed)
        elif op == "regex":
            bad = s.notna() & ~s.astype(str).str.fullmatch(val)
        else:
            return {"rule": text, "status": "ERROR", "failed": 0, "pct": 0.0, "note": "unknown operator"}
    except Exception as e:  # bad value typed by the user
        return {"rule": text, "status": "ERROR", "failed": 0, "pct": 0.0, "note": str(e)[:80]}
    failed = int(bad.sum())
    return {"rule": text, "status": "PASS" if failed == 0 else "FAIL", "failed": failed,
            "pct": round(failed / max(len(df), 1) * 100, 2), "note": ""}


def default_rules(df, prof, max_rules=30):
    """Pre-filled rules: completeness for problem/key columns, sensible numeric bounds,
    and allowed-value lists for low-cardinality text columns (guards against future bad loads)."""
    rules = []
    for c, i in prof["columns"].items():
        s = df[c]
        toks = _tokens(c)
        keyish = i["id_like"] or i["unique"] <= 30 or "code" in toks or "year" in toks
        if i["missing_pct"] > 0 or keyish:
            rules.append({"column": c, "op": "not_null", "value": ""})
        if is_numeric_dtype(s) and not is_bool_dtype(s):
            if "year" in toks:
                rules.append({"column": c, "op": "between", "value": "1900,2100"})
            elif expects_positive(c) and not i["id_like"]:
                rules.append({"column": c, "op": ">", "value": "0"})
        elif _is_text(s) and 2 <= i["unique"] <= 15:
            vals = [str(v) for v in s.dropna().unique() if str(v).strip().lower() not in NULL_TOKENS]
            if vals and not any("," in v for v in vals):
                rules.append({"column": c, "op": "in_set", "value": ",".join(sorted(vals))})
    return rules[:max_rules]


def rules_to_great_expectations(rule_list, suite_name="data_quality_suite"):
    """Export the current rules as a Great Expectations expectation-suite JSON string."""
    exps = []
    for r in rule_list:
        col, op, val = r["column"], r["op"], str(r.get("value", "") or "").strip()
        try:
            if op == "not_null":
                e = ("expect_column_values_to_not_be_null", {"column": col})
            elif op == "unique":
                e = ("expect_column_values_to_be_unique", {"column": col})
            elif op in (">", ">="):
                e = ("expect_column_values_to_be_between",
                     {"column": col, "min_value": float(val), "strict_min": op == ">"})
            elif op in ("<", "<="):
                e = ("expect_column_values_to_be_between",
                     {"column": col, "max_value": float(val), "strict_max": op == "<"})
            elif op == "between":
                lo, hi = [float(t) for t in val.split(",")]
                e = ("expect_column_values_to_be_between", {"column": col, "min_value": lo, "max_value": hi})
            elif op == "in_set":
                e = ("expect_column_values_to_be_in_set",
                     {"column": col, "value_set": [t.strip() for t in val.split(",")]})
            elif op == "regex":
                e = ("expect_column_values_to_match_regex", {"column": col, "regex": val})
            else:
                continue
        except ValueError:
            continue
        exps.append({"expectation_type": e[0], "kwargs": e[1], "meta": {}})
    return json.dumps({"expectation_suite_name": suite_name, "expectations": exps, "meta": {}}, indent=2)


# ----------------------------------------------------------------------
# DRIFT (schema + distribution)
# ----------------------------------------------------------------------
def _psi(base, cur, bins=10):
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate, >0.25 significant."""
    b = pd.to_numeric(base, errors="coerce").dropna()
    c = pd.to_numeric(cur, errors="coerce").dropna()
    if len(b) < 30 or len(c) < 30 or b.nunique() < 2:
        return None
    edges = np.unique(np.quantile(b, np.linspace(0, 1, bins + 1))).astype(float)
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    pb = np.clip(np.histogram(b, edges)[0] / len(b), 1e-4, None)
    pc = np.clip(np.histogram(c, edges)[0] / len(c), 1e-4, None)
    return float(np.sum((pc - pb) * np.log(pc / pb)))


def detect_drift(df, baseline):
    out = {"added": [], "removed": [], "dtype_changed": [], "numeric_shift": [],
           "category_shift": [], "has_baseline": baseline is not None}
    if baseline is None:
        return out
    cur, base = set(df.columns), set(baseline.columns)
    out["added"] = sorted(cur - base)
    out["removed"] = sorted(base - cur)
    for c in sorted(cur & base):
        if str(df[c].dtype) != str(baseline[c].dtype):
            out["dtype_changed"].append({"column": c, "was": str(baseline[c].dtype),
                                         "now": str(df[c].dtype)})
        if is_numeric_dtype(df[c]) and is_numeric_dtype(baseline[c]) and not is_id_like(c, df[c]):
            psi = _psi(baseline[c], df[c])
            if psi is not None and psi >= 0.1:
                out["numeric_shift"].append({"column": c, "psi": round(psi, 3),
                                             "level": "significant" if psi >= 0.25 else "moderate"})
        elif _is_text(df[c]) and _is_text(baseline[c]):
            if df[c].nunique() <= 50 and baseline[c].nunique() <= 50:
                pb = baseline[c].astype(str).value_counts(normalize=True)
                pc = df[c].astype(str).value_counts(normalize=True)
                idx = pb.index.union(pc.index)
                tvd = float(0.5 * (pc.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum())
                new_vals = sorted(set(pc.index) - set(pb.index))[:5]
                gone_vals = sorted(set(pb.index) - set(pc.index))[:5]
                if tvd >= 0.1 or new_vals or gone_vals:
                    out["category_shift"].append({"column": c, "tvd": round(tvd, 3),
                                                  "new_values": new_vals, "missing_values": gone_vals})
    return out


# ----------------------------------------------------------------------
# ETL LOG PARSING
# ----------------------------------------------------------------------
LOG_PATTERNS = [
    ("Schema change", r"unknown column|no such column|column .{0,40}(not found|missing|does not exist)|unexpected column|keyerror|schema"),
    ("Duplicate key", r"duplicate|unique constraint|primary key|integrity"),
    ("Type / format mismatch", r"type ?error|\bcast\b|could not convert|invalid literal|datatype|data type|parse|format"),
    ("Null / missing value", r"\bnull\b|\bnone\b|\bnan\b|cannot be null|missing"),
    ("Connection / timeout", r"timeout|timed out|connection (refused|reset|lost)|unreachable|deadlock"),
    ("Permission / auth", r"permission|denied|unauthori[sz]ed"),
    ("Encoding", r"encoding|codec|unicode|decode|utf"),
    ("Out of range / invalid value", r"out of range|overflow|invalid value|negative|check violation|constraint"),
    ("Resource / memory", r"memory|disk full|no space|\boom\b"),
]
_LEVEL = re.compile(r"\b(ERROR|FATAL|CRITICAL|WARN(?:ING)?|EXCEPTION)\b|Traceback", re.I)
_TS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")


def analyze_logs(text, columns=()):
    res = {"provided": bool(text and text.strip()), "lines": 0, "errors": 0,
           "warnings": 0, "categories": {}, "columns_mentioned": [],
           "first_ts": None, "last_ts": None}
    if not res["provided"]:
        return res
    lines = [l for l in text.splitlines() if l.strip()]
    res["lines"] = len(lines)
    ts = _TS.findall(text)
    if ts:
        res["first_ts"], res["last_ts"] = ts[0], ts[-1]
    for line in lines:
        m = _LEVEL.search(line)
        if not m:
            continue
        lvl = (m.group(1) or "ERROR").upper()
        res["warnings" if lvl.startswith("WARN") else "errors"] += 1
        cat = "Other"
        for name, pat in LOG_PATTERNS:
            if re.search(pat, line, re.I):
                cat = name
                break
        d = res["categories"].setdefault(cat, {"count": 0, "examples": []})
        d["count"] += 1
        if len(d["examples"]) < 2:
            d["examples"].append(line.strip()[:160])
    res["columns_mentioned"] = [c for c in columns if re.search(rf"\b{re.escape(str(c))}\b", text)]
    return res


# ----------------------------------------------------------------------
# CLEANING PLAN  ->  reviewable steps -> pandas code / SQL / cleaned dataframe
# ----------------------------------------------------------------------
def _best_predictor(df, col, numeric):
    y = df[col]
    mask = y.notna()
    n = int(mask.sum())
    if n < 30:
        return None
    best, best_score = None, 0.85
    for p in df.columns:
        if p == col:
            continue
        x = df[p]
        if not (_is_text(x) or is_id_like(p, x)):
            continue
        u = x.nunique()
        if u < 2 or u > n / 3:
            continue
        xm, ym = x[mask], y[mask]
        sizes = ym.groupby(xm).transform("size")
        keep = sizes >= 2
        if keep.sum() / n < 0.6:
            continue
        xm, ym = xm[keep], ym[keep]
        if numeric:
            tot = ((ym - ym.mean()) ** 2).sum()
            if tot == 0:
                continue
            score = 1 - ((ym - ym.groupby(xm).transform("mean")) ** 2).sum() / tot
        else:
            top = ym.groupby(xm).agg(lambda v: v.value_counts().iloc[0])
            score = top.sum() / len(ym)
        if score > best_score:
            best, best_score = {"by": p, "strength": round(float(score), 2)}, score
    return best


def build_plan(df, prof):
    cols = prof["columns"]
    plan = {"drop_duplicates": prof["duplicates"] > 0, "hidden_null_cols": {},
            "to_numeric": [], "label_maps": {}, "label_review": {},
            "zero_cols": [], "impute": [], "cap": []}
    for c, i in cols.items():
        if i["hidden_nulls"]:
            plan["hidden_null_cols"][c] = i["hidden_tokens"]
        if i["numeric_text"]:
            plan["to_numeric"].append(c)
        if i["label_groups"]:
            plan["label_maps"][c] = {v: g["canonical"] for g in i["label_groups"] for v in g["variants"]}
        if i["label_review"]:
            plan["label_review"][c] = {v: g["canonical"] for g in i["label_review"] for v in g["variants"]}
        if i["invalid_zeros"]:
            plan["zero_cols"].append(c)
        if i["outliers"]:
            plan["cap"].append(c)
    work = df.copy()
    for c, toks in plan["hidden_null_cols"].items():
        work[c] = work[c].where(~work[c].astype(str).str.strip().isin(toks))
    for c in plan["to_numeric"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    for c in plan["zero_cols"]:
        work.loc[work[c] <= 0, c] = np.nan
    for c in work.columns:
        if work[c].isna().any():
            numeric = is_numeric_dtype(work[c]) and not is_id_like(c, work[c])
            if is_id_like(c, work[c]) and is_numeric_dtype(work[c]):
                continue  # never invent IDs or years
            plan["impute"].append({"col": c, "numeric": bool(numeric),
                                   "by": _best_predictor(work, c, numeric)})
    return plan


def plan_steps(plan, prof):
    """Flat, human-reviewable list of every fix. Risky ones default to unapproved."""
    cols = prof["columns"]
    steps = []

    def add(id_, fix, col, detail, rows, default=True):
        steps.append({"id": id_, "fix": fix, "column": col, "detail": detail[:140],
                      "rows": int(rows), "default": default})

    if plan["drop_duplicates"]:
        add("dup", "Remove duplicates", "(all)", "Drop exact duplicate rows", prof["duplicates"])
    for c, toks in plan["hidden_null_cols"].items():
        add(f"hidden:{c}", "Hidden nulls -> NULL", c, f"Replace {toks} with NULL", cols[c]["hidden_nulls"])
    for c in plan["to_numeric"]:
        add(f"numeric:{c}", "Text -> number", c, "Convert to numeric; unparseable values become NULL",
            cols[c]["bad_numeric_text"])
    for c, m in plan["label_maps"].items():
        add(f"labels:{c}", "Merge label spellings", c, "; ".join(f"'{v}' -> '{k}'" for v, k in m.items()),
            cols[c]["nonstandard_rows"])
    for c, m in plan.get("label_review", {}).items():
        add(f"labels_review:{c}", "Merge POSSIBLE variants (review)", c,
            "; ".join(f"'{v}' -> '{k}'" for v, k in m.items()), cols[c]["review_rows"], default=False)
    for c in plan["zero_cols"]:
        add(f"zero:{c}", "Invalid zeros -> NULL", c, "0 / negative is not valid for this field",
            cols[c]["invalid_zeros"])
    for s in plan["impute"]:
        c = s["col"]
        how = (f"Fill from {s['by']['by']} ({s['by']['strength']:.0%} explained)" if s["by"]
               else ("Fill with median" if s["numeric"] else "Fill with 'Not Specified'"))
        add(f"impute:{c}", "Fill missing values", c, how,
            cols[c]["missing"] + cols[c]["hidden_nulls"] + cols[c]["invalid_zeros"])
    for c in plan["cap"]:
        add(f"cap:{c}", "Cap outliers (IQR)", c, "Clip to 1.5xIQR; some outliers may be genuine",
            cols[c]["outliers"], default=False)
    return steps


def filter_plan(plan, approved):
    """Return a plan containing only the approved steps."""
    ok = set(approved)
    p = {
        "drop_duplicates": bool(plan["drop_duplicates"] and "dup" in ok),
        "hidden_null_cols": {c: t for c, t in plan["hidden_null_cols"].items() if f"hidden:{c}" in ok},
        "to_numeric": [c for c in plan["to_numeric"] if f"numeric:{c}" in ok],
        "label_maps": {c: dict(m) for c, m in plan["label_maps"].items() if f"labels:{c}" in ok},
        "label_review": {c: m for c, m in plan.get("label_review", {}).items()
                         if f"labels_review:{c}" not in ok},      # still pending review
        "zero_cols": [c for c in plan["zero_cols"] if f"zero:{c}" in ok],
        "impute": [s for s in plan["impute"] if f"impute:{s['col']}" in ok],
        "cap": [c for c in plan["cap"] if f"cap:{c}" in ok],
    }
    for c, m in plan.get("label_review", {}).items():
        if f"labels_review:{c}" in ok:
            p["label_maps"].setdefault(c, {}).update(m)
    return p


def apply_plan(df, plan):
    d = df.copy()
    if plan["drop_duplicates"]:
        d = d.drop_duplicates()
    for c, toks in plan["hidden_null_cols"].items():
        d[c] = d[c].where(~d[c].astype(str).str.strip().isin(toks))
    for c in plan["to_numeric"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c, m in plan["label_maps"].items():
        d[c] = d[c].replace(m)
    for c in plan["zero_cols"]:
        d.loc[d[c] <= 0, c] = np.nan
    for step in plan["impute"]:
        c, by = step["col"], step["by"]["by"] if step["by"] else None
        if step["numeric"]:
            if by:
                d[c] = d[c].fillna(d.groupby(by)[c].transform("mean"))
            d[c] = d[c].fillna(d[c].median())
        else:
            if by:
                m = d.groupby(by)[c].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
                d[c] = d[c].fillna(d[by].map(m))
            d[c] = d[c].fillna("Not Specified")
    for c in plan["cap"]:
        q1, q3 = d[c].quantile(.25), d[c].quantile(.75)
        iqr = q3 - q1
        d[c] = d[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return d


def plan_to_pandas(plan):
    L = ["import pandas as pd", "import numpy as np", "",
         "def clean_dataset(df):", "    d = df.copy()"]
    n0 = len(L)
    if plan["drop_duplicates"]:
        L += ["    # 1. remove exact duplicate rows", "    d = d.drop_duplicates()"]
    for c, toks in plan["hidden_null_cols"].items():
        L += [f"    # hidden nulls in {c}: placeholder text standing in for NaN",
              f"    d[{c!r}] = d[{c!r}].where(~d[{c!r}].astype(str).str.strip().isin({toks!r}))"]
    for c in plan["to_numeric"]:
        L += [f"    d[{c!r}] = pd.to_numeric(d[{c!r}], errors='coerce')  # numbers stored as text"]
    for c, m in plan["label_maps"].items():
        L += [f"    # inconsistent labels in {c}", f"    d[{c!r}] = d[{c!r}].replace({m!r})"]
    for c in plan["zero_cols"]:
        L += [f"    d.loc[d[{c!r}] <= 0, {c!r}] = np.nan  # 0 is not a valid value here"]
    for s in plan["impute"]:
        c, by = s["col"], s["by"]["by"] if s["by"] else None
        L.append(f"    # impute {c}" + (f" using {by} (explains {s['by']['strength']:.0%})" if by else ""))
        if s["numeric"]:
            if by:
                L.append(f"    d[{c!r}] = d[{c!r}].fillna(d.groupby({by!r})[{c!r}].transform('mean'))")
            L.append(f"    d[{c!r}] = d[{c!r}].fillna(d[{c!r}].median())")
        else:
            if by:
                L += [f"    m = d.groupby({by!r})[{c!r}].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)",
                      f"    d[{c!r}] = d[{c!r}].fillna(d[{by!r}].map(m))"]
            L.append(f"    d[{c!r}] = d[{c!r}].fillna('Not Specified')")
    for c in plan["cap"]:
        L += [f"    q1, q3 = d[{c!r}].quantile(.25), d[{c!r}].quantile(.75)",
              f"    d[{c!r}] = d[{c!r}].clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))  # cap outliers in {c}"]
    if len(L) == n0:
        L.append("    # no approved fixes: dataset is returned unchanged")
    L.append("    return d")
    return "\n".join(L)


def plan_to_sql(plan, table="your_table"):
    q = []
    if plan["drop_duplicates"]:
        q.append(f"CREATE TABLE {table}_dedup AS SELECT DISTINCT * FROM {table};  -- then swap tables")
    for c, toks in plan["hidden_null_cols"].items():
        lst = ", ".join("'" + t.lower().replace("'", "''") + "'" for t in toks)
        q.append(f"UPDATE {table} SET {c} = NULL WHERE LOWER(TRIM({c})) IN ({lst});")
    for c, m in plan["label_maps"].items():
        by_canon = {}
        for v, canon in m.items():
            by_canon.setdefault(canon, []).append(v)
        for canon, vs in by_canon.items():
            lst = ", ".join("'" + v.replace("'", "''") + "'" for v in vs)
            q.append(f"UPDATE {table} SET {c} = '{canon}' WHERE {c} IN ({lst});")
    for c in plan["zero_cols"]:
        q.append(f"UPDATE {table} SET {c} = NULL WHERE {c} <= 0;")
    for s in plan["impute"]:
        c, by = s["col"], s["by"]["by"] if s["by"] else None
        if s["numeric"] and by:
            q.append(f"UPDATE {table} t SET {c} = (SELECT AVG(x.{c}) FROM {table} x WHERE x.{by} = t.{by}) "
                     f"WHERE t.{c} IS NULL;  -- MySQL: wrap the subquery in a derived table")
        elif not s["numeric"] and by:
            q.append(f"UPDATE {table} t SET {c} = (SELECT x.{c} FROM {table} x WHERE x.{by} = t.{by} AND x.{c} IS NOT NULL "
                     f"GROUP BY x.{c} ORDER BY COUNT(*) DESC LIMIT 1) WHERE t.{c} IS NULL;")
        if s["numeric"]:
            q.append(f"-- fallback for rows still NULL: SELECT COALESCE({c}, <median>) in the final query")
        else:
            q.append(f"UPDATE {table} SET {c} = 'Not Specified' WHERE {c} IS NULL;")
    for c in plan["cap"]:
        q.append(f"-- cap outliers in {c}: compute Q1/Q3 with PERCENTILE_CONT(0.25/0.75), then "
                 f"SELECT LEAST(GREATEST({c}, q1-1.5*(q3-q1)), q3+1.5*(q3-q1))")
    return "\n".join(q) if q else "-- no approved SQL fixes"