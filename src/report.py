"""Evidence-based report builder. Every statement is derived from the profile,
rule results, drift check or log analysis; sections with no evidence are skipped."""
from src.dq_engine import plan_to_pandas, plan_to_sql


def _business_summary(prof, rules, drift, logs, plan):
    s = prof["score"]
    cols = prof["columns"]
    worst = sorted(cols.items(), key=lambda kv: -kv[1]["missing_pct"])
    lines = [f"- Overall data health is **{s['label']} ({s['value']}/100)**. "
             f"The score counts gaps, duplicates and invalid values, and penalises any column that is more than 5% empty."]
    gaps = [(c, i) for c, i in worst if i["missing_pct"] > 5]
    if gaps:
        txt = ", ".join(f"**{c}** ({i['missing_pct']}% empty)" for c, i in gaps)
        lines.append(f"- Reports that use {txt} are working from incomplete information, "
                     f"so totals and averages for those fields may be misleading.")
    hidden = [c for c, i in cols.items() if i["hidden_nulls"] or i["invalid_zeros"]]
    if hidden:
        lines.append(f"- Some values look filled in but are really placeholders ({', '.join(hidden)}). "
                     f"They hide missing data from simple null checks.")
    lab = [c for c, i in cols.items() if i["label_groups"]]
    if lab:
        lines.append(f"- The same category is spelled several ways in {', '.join(lab)}, "
                     f"so grouping or counting by that field splits one category into several.")
    conf = [c for c, i in cols.items() if i["scale_conflicts"]]
    if conf:
        lines.append(f"- **{', '.join(conf)}** contain labels that differ only by a scale or unit word "
                     f"(for example 'Dollars' vs 'Dollars (millions)'). These are **not** merged automatically: "
                     f"they are probably different units, and merging them would change values by orders of magnitude.")
    rev = [c for c, i in cols.items() if i["label_review"]]
    if rev:
        lines.append(f"- Possible label variants in {', '.join(rev)} need a human decision before merging.")
    if drift["has_baseline"]:
        shifted = len(drift["numeric_shift"]) + len(drift["category_shift"])
        if shifted or drift["added"] or drift["removed"] or drift["dtype_changed"]:
            lines.append("- Compared with the baseline file, the schema or the value distributions have changed.")
        else:
            lines.append("- The data matches the baseline file in schema and distribution.")
    if not (gaps or hidden or lab or conf or rev):
        lines.append("- No significant completeness or consistency problems were found.")
    return "\n".join(lines)


def _issue_items(prof, rules, drift):
    """Return (severity, text) pairs. Severity: Critical / Warning / Info."""
    N = max(prof["rows"], 1)
    items = []
    for c, i in prof["columns"].items():
        if i["missing"]:
            sev = "Critical" if i["missing_pct"] > 20 else "Warning" if i["missing_pct"] > 5 else "Info"
            items.append((sev, f"**NULLs - {c}:** {i['missing']:,} missing ({i['missing_pct']}%)."))
        if i["hidden_nulls"]:
            pct = i["hidden_nulls"] / N * 100
            items.append(("Critical" if pct > 5 else "Warning",
                          f"**Hidden nulls - {c}:** {i['hidden_nulls']:,} placeholder values {i['hidden_tokens']} ({pct:.1f}%)."))
        if i["invalid_zeros"]:
            pct = i["invalid_zeros"] / N * 100
            items.append(("Critical" if pct > 5 else "Warning",
                          f"**Invalid values - {c}:** {i['invalid_zeros']:,} zero/negative values ({pct:.1f}%) "
                          f"in a field that should be positive."))
        if i["bad_numeric_text"]:
            items.append(("Warning", f"**Format - {c}:** numeric column contains {i['bad_numeric_text']:,} non-numeric text values."))
        for g in i["label_groups"]:
            items.append(("Warning", f"**Inconsistent labels - {c}:** {g['variants']} look like variants of '{g['canonical']}' "
                                     f"(only case/spacing differs; safe to merge)."))
        for g in i["label_review"]:
            items.append(("Warning", f"**Possible label variants - {c}:** {g['variants']} may be the same as '{g['canonical']}'. "
                                     f"**Review before merging.**"))
        for a, b in i["scale_conflicts"]:
            items.append(("Critical", f"**Mixed units - {c}:** '{a}' and '{b}' differ by a scale word. They are probably "
                                      f"different units, not typos. Do **not** merge; convert values explicitly or split the column."))
        if i["outliers"]:
            items.append(("Info", f"**Outliers - {c}:** {i['outliers']:,} values outside 1.5xIQR "
                                  f"({i['outliers'] / N * 100:.1f}%). Review before capping; some may be genuine."))
    if prof["duplicates"]:
        items.append(("Warning", f"**Duplicates:** {prof['duplicates']:,} exact duplicate rows."))
    for r in rules:
        if r["status"] == "FAIL":
            items.append(("Critical" if r["pct"] > 5 else "Warning",
                          f"**Rule failed:** `{r['rule']}` - {r['failed']:,} rows ({r['pct']}%)."))
        elif r["status"] == "ERROR":
            items.append(("Info", f"**Rule error:** `{r['rule']}` - {r['note']}."))
    if drift["has_baseline"]:
        for c in drift["removed"]:
            items.append(("Critical", f"**Schema drift:** column '{c}' is in the baseline but missing now."))
        for c in drift["added"]:
            items.append(("Warning", f"**Schema drift:** new column '{c}' not in the baseline."))
        for d in drift["dtype_changed"]:
            items.append(("Warning", f"**Schema drift:** '{d['column']}' type changed {d['was']} -> {d['now']}."))
        for d in drift["numeric_shift"]:
            items.append(("Critical" if d["level"] == "significant" else "Warning",
                          f"**Distribution drift - {d['column']}:** PSI {d['psi']} ({d['level']} shift vs baseline)."))
        for d in drift["category_shift"]:
            bits = []
            if d["new_values"]:
                bits.append(f"new values {d['new_values']}")
            if d["missing_values"]:
                bits.append(f"values gone {d['missing_values']}")
            sev = "Critical" if d["tvd"] >= 0.25 else "Warning"
            items.append((sev, f"**Category drift - {d['column']}:** share shifted by {d['tvd'] * 100:.0f}%"
                               + (f"; {', '.join(bits)}" if bits else "") + "."))
    return items


def _issues(prof, rules, drift):
    L = [f"**Volume:** {prof['rows']:,} rows, {prof['columns_count']} columns, {prof['duplicates']} duplicate rows.", ""]
    items = _issue_items(prof, rules, drift)
    for sev, icon in (("Critical", "🔴"), ("Warning", "🟠"), ("Info", "🔵")):
        sub = [t for s, t in items if s == sev]
        if sub:
            L.append(f"**{icon} {sev} ({len(sub)})**")
            L += [f"- {t}" for t in sub]
            L.append("")
    if rules:
        fails = [r for r in rules if r["status"] == "FAIL"]
        L.append(f"**Rule checks:** {len(rules) - len(fails)} of {len(rules)} rules passed.")
    if drift["has_baseline"] and not (drift["added"] or drift["removed"] or drift["dtype_changed"]
                                      or drift["numeric_shift"] or drift["category_shift"]):
        L.append("**Drift:** none - schema and distributions match the baseline.")
    if not items:
        L.append("No data-quality issues were found.")
    return "\n".join(L)


def _root_causes(prof, drift, logs, plan):
    L = []
    imp = {s["col"]: s for s in plan["impute"]}
    for c, i in prof["columns"].items():
        if i["missing_pct"] > 5:
            s = imp.get(c)
            if s and s["by"]:
                L.append(f"- **{c} ({i['missing_pct']}% missing):** the value is recoverable from **{s['by']['by']}** "
                         f"(explains ~{s['by']['strength']:.0%} of it). Likely cause: the field was not populated on some "
                         f"source records or was dropped in a join/merge, even though other rows carry it.")
            else:
                L.append(f"- **{c} ({i['missing_pct']}% missing):** no other column predicts it. Likely a field "
                         f"not captured at the source or lost during extraction.")
        if i["invalid_zeros"] and i["hidden_nulls"] == 0:
            L.append(f"- **{c}:** zeros in a positive-only field usually mean the source system or ETL "
                     f"defaulted 'unknown' to 0.")
        if i["label_groups"] or i["label_review"]:
            L.append(f"- **{c}:** mixed spellings suggest free-text entry or several source systems with "
                     f"different code lists and no standardisation step.")
        if i["scale_conflicts"]:
            L.append(f"- **{c}:** both a plain and a scaled unit label exist. Typical causes are a file that "
                     f"mixes two publishing conventions, or a pipeline that dropped the scale suffix. "
                     f"Check the source documentation for which rows are in which unit.")
        if i["hidden_nulls"]:
            L.append(f"- **{c}:** placeholder strings suggest the source writes text like 'N/A' instead of NULL.")
    if drift["has_baseline"] and (drift["added"] or drift["removed"] or drift["dtype_changed"]):
        L.append("- **Schema drift:** upstream table or file layout changed since the baseline; "
                 "mappings and type casts in the pipeline may no longer match.")
    if drift["has_baseline"] and (drift["numeric_shift"] or drift["category_shift"]):
        L.append("- **Distribution drift:** values moved even though the schema did not. Typical causes are a changed "
                 "business rule, a new source feeding the table, or a unit/scale change upstream.")
    if logs["provided"] and logs["categories"]:
        for cat, d in sorted(logs["categories"].items(), key=lambda kv: -kv[1]["count"]):
            L.append(f"- **Logs - {cat} ({d['count']} events):** e.g. `{d['examples'][0]}`")
        if logs["columns_mentioned"]:
            L.append(f"- **Log <-> data link:** the logs mention dataset columns {logs['columns_mentioned']}.")
    if not L:
        L.append("- No root causes to report: nothing above the reporting thresholds.")
    L.append("\n*These are hypotheses based on patterns in the data; confirm against the source system.*")
    return "\n".join(L)


def _log_section(logs):
    if not logs["provided"]:
        return None
    if not logs["categories"]:
        return f"{logs['lines']} log lines were read; no ERROR/WARN entries were found."
    return (f"{logs['errors']} errors and {logs['warnings']} warnings in {logs['lines']} lines"
            + (f" ({logs['first_ts']} -> {logs['last_ts']})." if logs["first_ts"] else "."))


def build_report(prof, rules, drift, logs, plan, applied_plan=None):
    """`plan` is the full detected plan (for root causes); `applied_plan` is what the user approved
    (defaults to the full plan) and is what the generated code reflects."""
    applied = applied_plan if applied_plan is not None else plan
    parts = ["## 1. Business summary", _business_summary(prof, rules, drift, logs, plan),
             "## 2. Identified data-quality issues", _issues(prof, rules, drift)]
    ls = _log_section(logs)
    parts += ["## 3. ETL log analysis",
              ls if ls else "_No ETL logs were uploaded, so no pipeline-failure analysis was done._"]
    parts += ["## 4. Possible root causes", _root_causes(prof, drift, logs, plan),
              "## 5. Corrective code (Python / Pandas)", "```python\n" + plan_to_pandas(applied) + "\n```",
              "## 6. Corrective code (SQL)", "```sql\n" + plan_to_sql(applied) + "\n```",
              "## 7. Prevention recommendations", _prevention(prof, plan, drift, logs)]
    return "\n\n".join(parts)


def _prevention(prof, plan, drift, logs):
    R = ["- Add a **not-null / completeness check** at ingestion for: "
         + (", ".join(c for c, i in prof["columns"].items() if i["missing_pct"] > 5) or "critical fields") + "."]
    if plan["hidden_null_cols"] or plan["zero_cols"]:
        R.append("- Convert placeholders ('N/A', 'None', 0 as unknown) to real NULLs in the extract step.")
    allowed = sorted(set(plan["label_maps"]) | set(plan.get("label_review", {}))
                     | {c for c, i in prof["columns"].items() if i["scale_conflicts"]})
    if allowed:
        R.append("- Enforce an **allowed-values list** (reference table or enum) for: " + ", ".join(allowed) + ".")
    if any(i["scale_conflicts"] for i in prof["columns"].values()):
        R.append("- Store the unit and the scale (units / thousands / millions) in **separate** fields, "
                 "so a scale change can never be mistaken for a spelling change.")
    R.append("- Validate schema (names + types) on every load with Great Expectations or Pydantic; fail fast on drift. "
             "The app can export your current rules as a Great Expectations suite.")
    if prof["duplicates"]:
        R.append("- Add a primary/unique key constraint or a dedup step before loading.")
    if logs["provided"]:
        R.append("- Alert on the recurring log categories above instead of reading logs manually.")
    R.append("- Track the quality score per load and alert when it drops.")
    return "\n".join(R)