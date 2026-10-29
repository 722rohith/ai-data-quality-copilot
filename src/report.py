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
    if not (gaps or hidden or lab):
        lines.append("- No significant completeness or consistency problems were found.")
    return "\n".join(lines)


def _issues(prof, rules, drift):
    L = [f"- **Volume & duplicates:** {prof['rows']:,} rows, {prof['columns_count']} columns, "
         f"{prof['duplicates']} duplicate rows."]
    for c, i in prof["columns"].items():
        if i["missing"]:
            L.append(f"- **NULLs – {c}:** {i['missing']:,} missing ({i['missing_pct']}%).")
        if i["hidden_nulls"]:
            L.append(f"- **Hidden nulls – {c}:** {i['hidden_nulls']:,} placeholder values {i['hidden_tokens']}.")
        if i["invalid_zeros"]:
            L.append(f"- **Invalid values – {c}:** {i['invalid_zeros']:,} zero/negative values "
                     f"({i['invalid_zeros'] / prof['rows'] * 100:.1f}%) in a field that should be positive.")
        if i["bad_numeric_text"]:
            L.append(f"- **Format – {c}:** numeric column contains {i['bad_numeric_text']:,} non-numeric text values.")
        for g in i["label_groups"]:
            L.append(f"- **Inconsistent labels – {c}:** {g['variants']} look like variants of '{g['canonical']}'.")
        if i["outliers"]:
            L.append(f"- **Outliers – {c}:** {i['outliers']:,} values outside 1.5×IQR "
                     f"({i['outliers'] / prof['rows'] * 100:.1f}%). Review before capping; some may be genuine.")
    fails = [r for r in rules if r["status"] == "FAIL"]
    if rules:
        L.append(f"- **Rule checks:** {len(rules) - len(fails)} of {len(rules)} rules passed.")
    if drift["has_baseline"]:
        if drift["added"] or drift["removed"] or drift["dtype_changed"]:
            L.append(f"- **Schema drift:** added {drift['added'] or 'none'}, removed {drift['removed'] or 'none'}, "
                     f"type changes {[(d['column'], d['was'], d['now']) for d in drift['dtype_changed']] or 'none'}.")
        else:
            L.append("- **Schema drift:** none – columns and types match the baseline.")
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
        if i["label_groups"]:
            L.append(f"- **{c}:** mixed spellings suggest free-text entry or several source systems with "
                     f"different code lists and no standardisation step.")
        if i["hidden_nulls"]:
            L.append(f"- **{c}:** placeholder strings suggest the source writes text like 'N/A' instead of NULL.")
    if drift["has_baseline"] and (drift["added"] or drift["removed"] or drift["dtype_changed"]):
        L.append("- **Schema drift:** upstream table or file layout changed since the baseline; "
                 "mappings and type casts in the pipeline may no longer match.")
    if logs["provided"] and logs["categories"]:
        for cat, d in sorted(logs["categories"].items(), key=lambda kv: -kv[1]["count"]):
            L.append(f"- **Logs – {cat} ({d['count']} events):** e.g. `{d['examples'][0]}`")
        if logs["columns_mentioned"]:
            L.append(f"- **Log ↔ data link:** the logs mention dataset columns {logs['columns_mentioned']}.")
    if not L:
        L.append("- No root causes to report: nothing above the reporting thresholds.")
    L.append("\n*These are hypotheses based on patterns in the data; confirm against the source system.*")
    return "\n".join(L)


def _log_section(logs):
    if not logs["provided"]:
        return None
    if not logs["categories"]:
        return f"{logs['lines']} log lines were read; no ERROR/WARN entries were found."
    head = (f"{logs['errors']} errors and {logs['warnings']} warnings in {logs['lines']} lines"
            + (f" ({logs['first_ts']} → {logs['last_ts']})." if logs["first_ts"] else "."))
    return head


def build_report(prof, rules, drift, logs, plan):
    parts = ["## 1. Business summary", _business_summary(prof, rules, drift, logs, plan),
             "## 2. Identified data-quality issues", _issues(prof, rules, drift)]
    ls = _log_section(logs)
    parts += ["## 3. ETL log analysis",
              ls if ls else "_No ETL logs were uploaded, so no pipeline-failure analysis was done._"]
    parts += ["## 4. Possible root causes", _root_causes(prof, drift, logs, plan),
              "## 5. Corrective code (Python / Pandas)", "```python\n" + plan_to_pandas(plan) + "\n```",
              "## 6. Corrective code (SQL)", "```sql\n" + plan_to_sql(plan) + "\n```",
              "## 7. Prevention recommendations", _prevention(prof, plan, drift, logs)]
    return "\n\n".join(parts)


def _prevention(prof, plan, drift, logs):
    R = ["- Add a **not-null / completeness check** at ingestion for: "
         + (", ".join(c for c, i in prof["columns"].items() if i["missing_pct"] > 5) or "critical fields") + "."]
    if plan["hidden_null_cols"] or plan["zero_cols"]:
        R.append("- Convert placeholders ('N/A', 'None', 0 as unknown) to real NULLs in the extract step.")
    if plan["label_maps"]:
        R.append("- Enforce an **allowed-values list** (reference table or enum) for: " + ", ".join(plan["label_maps"]) + ".")
    R.append("- Validate schema (names + types) on every load with Great Expectations or Pydantic; fail fast on drift.")
    if prof["duplicates"]:
        R.append("- Add a primary/unique key constraint or a dedup step before loading.")
    if logs["provided"]:
        R.append("- Alert on the recurring log categories above instead of reading logs manually.")
    R.append("- Track the quality score per load and alert when it drops.")
    return "\n".join(R)