import io
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.dq_engine import (profile_dataframe, default_rules, evaluate_rule, RULE_OPS,
                           detect_drift, analyze_logs, build_plan, apply_plan,
                           plan_steps, filter_plan, hidden_total, rules_to_great_expectations)
from src.report import build_report
from src.llm import answer as ask_copilot

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import markdown as md_lib
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

st.set_page_config(
    page_title="AI Data Quality & Root-Cause Copilot",
    page_icon="📋",
    layout="wide"
)

# ======================================================================
# PREMIUM ANIMATED BLACK & GOLD THEME
# ======================================================================
st.markdown("""
<style>
@keyframes slideUp  {from{opacity:0; transform:translateY(26px)} to{opacity:1; transform:translateY(0)}}
@keyframes fadeIn   {from{opacity:0} to{opacity:1}}
@keyframes shimmer  {0%{background-position:0% center} 100%{background-position:200% center}}
@keyframes floaty   {0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)}}
@keyframes shine    {from{transform:translateX(-130%)} to{transform:translateX(260%)}}
@keyframes lineMove {0%{background-position:0% 50%} 100%{background-position:200% 50%}}
@keyframes glow     {0%,100%{box-shadow:0 0 0 rgba(212,175,55,0)} 50%{box-shadow:0 0 22px rgba(212,175,55,.35)}}

.stApp {background: radial-gradient(circle at 20% 0%, #17140a 0%, #0c0c0e 45%); color:#f3f3f3;}
.block-container {max-width: 1250px; padding-top: 2rem; animation: fadeIn .6s ease both;}
h2, h3 {color:#d4af37 !important; font-weight:700;}

.page-hero {
    position: relative; overflow: hidden; display:flex; align-items:center; gap:22px;
    background: linear-gradient(135deg, rgba(212,175,55,.14), rgba(22,22,26,.9) 55%);
    border: 1px solid rgba(212,175,55,.45); border-radius: 20px;
    padding: 28px 34px; margin-bottom: 22px;
    box-shadow: 0 12px 34px rgba(212,175,55,.12);
    animation: slideUp .7s ease both;
}
.page-hero::after {
    content:""; position:absolute; top:0; left:0; width:30%; height:100%;
    background: linear-gradient(100deg, transparent, rgba(247,224,138,.14), transparent);
    animation: shine 6s ease-in-out infinite;
}
.ph-icon {font-size: 3.2rem; animation: floaty 3.2s ease-in-out infinite;}
.ph-title {
    margin:0; font-size: 1.9rem; font-weight: 800; letter-spacing:.02em;
    background: linear-gradient(90deg,#b8932a,#f7e08a,#d4af37,#f7e08a,#b8932a);
    background-size:200% auto; -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color: transparent; animation: shimmer 5s linear infinite;
}
.page-hero p {margin:6px 0 0 0; color:#b9b9c0; font-size:1rem;}

.section-title {
    font-size:1.25rem; font-weight:700; color:#d4af37; margin: 18px 0 10px 0;
    animation: slideUp .6s ease both;
}

[data-testid="stFileUploader"] section {
    background: rgba(22,22,26,.9); border: 1px dashed rgba(212,175,55,.6); border-radius: 14px;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stFileUploader"] section:hover {
    transform: translateY(-3px); border-color:#f7e08a; box-shadow: 0 10px 26px rgba(212,175,55,.25);
}
[data-testid="stFileUploader"] label p {color:#f3f3f3; font-weight:600;}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(212,175,55,.3); border-radius: 14px; animation: slideUp .8s ease both;
}
[data-testid="stPlotlyChart"] {
    background:#16161a; border:1px solid rgba(212,175,55,.25); border-radius:16px;
    padding:6px; animation: fadeIn 1s ease both; transition: transform .25s ease, box-shadow .25s ease;
}
[data-testid="stPlotlyChart"]:hover {transform: translateY(-4px); box-shadow:0 12px 30px rgba(212,175,55,.18);}

div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(90deg,#d4af37 0%,#aa8c2c 100%); color:#0c0c0e; font-weight:700;
    border:none; border-radius:10px; padding:0.7rem 1.4rem;
    box-shadow: 0 4px 14px rgba(212,175,55,.3); animation: glow 3s ease-in-out infinite;
    transition: transform .2s ease, box-shadow .2s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background: linear-gradient(90deg,#ffd700 0%,#d4af37 100%); color:#000;
    transform: translateY(-3px); box-shadow: 0 10px 24px rgba(212,175,55,.45);
}
div.stButton > button:active {transform: scale(.95) translateY(1px);}

.report-head {
    display:flex; align-items:center; gap:14px; padding:18px 24px; border-radius:16px;
    background: linear-gradient(135deg, rgba(212,175,55,.16), rgba(22,22,26,.95));
    border:1px solid rgba(212,175,55,.5); margin: 8px 0 16px 0; animation: slideUp .7s ease both;
}
.report-head .ri {font-size:2rem; animation: floaty 3s ease-in-out infinite;}
.report-head .rt {font-size:1.3rem; font-weight:800; color:#f7e08a;}
.report-head .rs {color:#b9b9c0; font-size:.9rem;}

hr {
    border:none; height:2px; background: linear-gradient(90deg, transparent,#d4af37,#f7e08a,#d4af37, transparent);
    background-size:200% 100%; animation: lineMove 6s linear infinite;
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# UI HELPERS
# ======================================================================
def page_header(icon, title, subtitle):
    st.markdown(
        f'<div class="page-hero"><div class="ph-icon">{icon}</div>'
        f'<div><h1 class="ph-title">{title}</h1><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True)


def section(icon, text):
    st.markdown(f'<div class="section-title">{icon} {text}</div>', unsafe_allow_html=True)


def kpi_cards(items, height=210):
    """Animated KPI cards with count-up numbers."""
    items = [
        {
            "label": str(it["label"]),
            "value": float(it["value"]),
            "prefix": str(it.get("prefix", "")),
            "suffix": str(it.get("suffix", "")),
            "decimals": int(it.get("decimals", 0)),
            "delta": None if it.get("delta") is None else str(it["delta"]),
            "positive": bool(it.get("positive", True)),
        }
        for it in items
    ]
    html = """
    <style>
    * {box-sizing:border-box;}
    body {margin:0; font-family:'Inter','Segoe UI',sans-serif; color:#f3f3f3; background:transparent;}
    .grid {display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; padding:6px 4px 14px 4px;}
    .card {
        background: linear-gradient(145deg, rgba(212,175,55,.12), rgba(22,22,26,.95));
        border:1px solid rgba(212,175,55,.55); border-radius:16px; padding:16px 16px;
        opacity:0; transform:translateY(26px); animation: up .7s ease forwards;
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    }
    .card:hover {transform:translateY(-8px); border-color:#f7e08a; box-shadow:0 14px 32px rgba(212,175,55,.3);}
    .label {font-size:.9rem; opacity:.8; margin-bottom:8px;}
    .value {font-size:1.6rem; font-weight:800; margin-bottom:10px; white-space:nowrap;}
    .pill {display:inline-block; font-size:.75rem; font-weight:600; padding:4px 10px; border-radius:999px;}
    .up {background:rgba(34,197,94,.15); color:#4ade80;}
    .down {background:rgba(239,68,68,.15); color:#f87171;}
    @keyframes up {to{opacity:1; transform:translateY(0)}}
    </style>
    <div class="grid" id="g"></div>
    <script>
    const items = __DATA__;
    const g = document.getElementById('g');
    g.style.gridTemplateColumns = 'repeat(' + items.length + ',minmax(0,1fr))';
    items.forEach((it, i) => {
        const c = document.createElement('div');
        c.className = 'card'; c.style.animationDelay = (i*0.12)+'s';
        let pill = '';
        if (it.delta) {
            const cls = it.positive ? 'up' : 'down';
            pill = '<span class="pill '+cls+'">'+(it.positive ? '✓ ' : '⚠ ')+it.delta+'</span>';
        }
        c.innerHTML = '<div class="label">'+it.label+'</div><div class="value" id="v'+i+'">0</div>'+pill;
        g.appendChild(c);
        const el = document.getElementById('v'+i);
        const fmt = v => (it.prefix||'') + v.toLocaleString('en-US',
            {minimumFractionDigits: it.decimals, maximumFractionDigits: it.decimals}) + (it.suffix ? '<span style="font-size:.8rem;opacity:.7">'+it.suffix+'</span>' : '');
        const start = performance.now() + i*120, dur = 1400;
        function tick(now) {
            const t = Math.max(0, Math.min(1, (now-start)/dur));
            const e = 1 - Math.pow(1-t, 3);
            el.innerHTML = fmt(it.value * e);
            if (t < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    });
    </script>
    """.replace("__DATA__", json.dumps(items))
    components.html(html, height=height)


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f3f3",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(gridcolor="rgba(212,175,55,.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(212,175,55,.10)", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def load_csv(data: bytes):
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def cached_profile(data: bytes):
    """Profile once per uploaded file instead of on every Streamlit rerun."""
    return profile_dataframe(load_csv(data))


def column_profile(df, cols, n):
    """Per-column profile table: type, nulls, distinct, range, top value, outliers."""
    rows = []
    for c in df.columns:
        s = df[c]
        info = cols.get(c, {})
        row = {
            "Column": c,
            "Type": str(s.dtype),
            "Missing %": round(s.isna().mean() * 100, 2),
            "Hidden/invalid %": round((info.get("hidden_nulls", 0) + info.get("invalid_zeros", 0)
                                       + info.get("bad_numeric_text", 0)) / n * 100, 2),
            "Label variants %": round((info.get("nonstandard_rows", 0) + info.get("review_rows", 0)) / n * 100, 2),
            "Distinct": int(s.nunique(dropna=True)),
            "Min": None, "Max": None, "Outliers (IQR)": None,
            "Top value": None,
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            row["Min"], row["Max"] = s.min(), s.max()
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            row["Outliers (IQR)"] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        else:
            vc = s.value_counts(dropna=True)
            if len(vc):
                row["Top value"] = f"{vc.index[0]} ({vc.iloc[0]:,})"
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ("Min", "Max"):
        out[c] = out[c].astype("float64")
    return out


def report_html(md_text, title):
    body = md_lib.markdown(md_text, extensions=["tables", "fenced_code"]) if HAS_MARKDOWN else f"<pre>{md_text}</pre>"
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:40px auto;color:#222;line-height:1.55}"
            "h1,h2,h3{color:#8a6d1a}code,pre{background:#f4f1e6;padding:2px 6px;border-radius:4px}"
            "pre{padding:12px;overflow:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
            f"</style></head><body>{body}</body></html>")


# ======================================================================
# PAGE
# ======================================================================
page_header("📋", "AI DATA QUALITY & ROOT-CAUSE COPILOT",
            "Upload your dataset and ETL error logs to generate an instant diagnostic and remediation report.")

up1, up2, up3 = st.columns(3)
with up1:
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], key="main_dataset")
with up2:
    error_log_file = st.file_uploader("Upload ETL Error Logs (Optional, TXT/LOG)", type=["txt", "log"], key="error_logs")
with up3:
    baseline_file = st.file_uploader("Baseline CSV (Optional, drift check)", type=["csv"], key="baseline_csv")

if error_log_file is not None or baseline_file is not None:
    info_msgs = []
    if error_log_file is not None:
        info_msgs.append("📄 ETL Error Logs uploaded.")
    if baseline_file is not None:
        info_msgs.append("📊 Baseline CSV uploaded.")
    if uploaded_file is None:
        st.info("ℹ️ " + " ".join(info_msgs) + " Please upload your **Dataset (CSV)** in the first box to run the analysis.")

if uploaded_file is not None:
    raw = uploaded_file.getvalue()
    df = load_csv(raw)
    baseline_df = load_csv(baseline_file.getvalue()) if baseline_file is not None else None
    error_logs = error_log_file.getvalue().decode("utf-8", errors="replace") if error_log_file is not None else ""

    prof = cached_profile(raw)
    score = prof["score"]
    cols = prof["columns"]
    n = max(prof["rows"], 1)
    hidden_cells = hidden_total(prof)
    review_cells = sum(i["review_rows"] for i in cols.values())
    scale_cols = [c for c, i in cols.items() if i["scale_conflicts"]]

    # ---------- KPI cards ----------
    section("🧮", "Data Health Overview")
    kpi_cards([
        {"label": "Total Rows", "value": prof["rows"], "decimals": 0},
        {"label": "Columns", "value": prof["columns_count"], "decimals": 0},
        {"label": "Missing Cells", "value": prof["missing_cells"], "decimals": 0,
         "delta": "Clean" if prof["missing_cells"] == 0 else f"{score['avg_missing']:.1f}% avg",
         "positive": prof["missing_cells"] == 0},
        {"label": "Hidden / Invalid", "value": hidden_cells, "decimals": 0,
         "delta": "None found" if hidden_cells == 0 else "hidden problems",
         "positive": hidden_cells == 0},
        {"label": "Duplicate Rows", "value": prof["duplicates"], "decimals": 0,
         "delta": "None found" if prof["duplicates"] == 0 else f"{score['duplicates']:.1f}%",
         "positive": prof["duplicates"] == 0},
        {"label": "Quality Score", "value": score["value"], "suffix": " / 100", "decimals": 1,
         "delta": score["label"], "positive": score["value"] >= 75},
    ])

    if scale_cols:
        st.warning("⚠️ **Mixed units detected in " + ", ".join(f"`{c}`" for c in scale_cols) + ".** "
                   + "; ".join(f"'{a}' vs '{b}'" for c in scale_cols for a, b in cols[c]["scale_conflicts"][:2])
                   + ". These look like different scales, so they are **never merged automatically**.")
    if review_cells:
        st.info(f"🔎 {review_cells:,} cells have label variants that need your review before merging (see the review step below).")

    # ---------- explorer ----------
    section("📊", "Dataset Explorer")
    tab_prev, tab_prof = st.tabs(["Preview", "Column profile"])
    with tab_prev:
        st.dataframe(df.head(10), use_container_width=True)
    with tab_prof:
        st.dataframe(column_profile(df, cols, n), use_container_width=True, hide_index=True)
        st.caption("Outliers use the 1.5×IQR rule. 'Hidden/invalid' = disguised nulls, invalid zeros and non-numeric text.")

    # ---------- charts ----------
    section("📈", "Data Quality Insights")
    ch1, ch2 = st.columns([3, 2])
    rows = []
    for c, i in cols.items():
        for label, val in (("Missing", i["missing"]),
                           ("Hidden nulls", i["hidden_nulls"] + i["bad_numeric_text"]),
                           ("Invalid zeros", i["invalid_zeros"]),
                           ("Label variants (safe)", i["nonstandard_rows"]),
                           ("Label variants (review)", i["review_rows"])):
            if val:
                rows.append({"Column": c, "Issue": label, "Affected %": round(val / n * 100, 2)})
    issue_df = pd.DataFrame(rows)

    with ch1:
        if issue_df.empty:
            st.success("✅ No missing, hidden or invalid values found in any column.")
        elif HAS_PLOTLY:
            fig_issue = px.bar(issue_df, x="Column", y="Affected %", color="Issue",
                               title="Affected Rows by Column and Issue Type (%)",
                               color_discrete_map={"Missing": "#ffd700", "Hidden nulls": "#d4af37",
                                                   "Invalid zeros": "#f87171",
                                                   "Label variants (safe)": "#60a5fa",
                                                   "Label variants (review)": "#a78bfa"})
            st.plotly_chart(style_fig(fig_issue), use_container_width=True)
        else:
            st.caption("Affected rows by column and issue type (%). Install `plotly` for the full chart.")
            st.bar_chart(issue_df.pivot_table(index="Column", columns="Issue", values="Affected %", fill_value=0),
                         y_label="Affected rows (%)")
            st.dataframe(issue_df, use_container_width=True, hide_index=True)

    with ch2:
        if HAS_PLOTLY:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score["value"],
                title={"text": f"Quality Score · {score['label']}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d4af37"},
                    "bgcolor": "rgba(255,255,255,.04)",
                    "steps": [
                        {"range": [0, 60], "color": "rgba(239,68,68,.25)"},
                        {"range": [60, 90], "color": "rgba(234,179,8,.25)"},
                        {"range": [90, 100], "color": "rgba(34,197,94,.25)"},
                    ],
                }))
            st.plotly_chart(style_fig(fig_gauge), use_container_width=True)
        else:
            st.metric("Quality Score", f"{score['value']:.1f} / 100")
        st.caption(f"Score = 100 − avg missing {score['avg_missing']} − duplicates {score['duplicates']} "
                   f"− invalid values {score['invalid']} − worst-column penalty {score['column_penalty']}")

    # ---------- rule checks ----------
    section("✅", "Data-Quality Rules")
    seed_rules = default_rules(df, prof)
    st.caption("Auto-generated rules are pre-filled from your data (completeness, ranges, allowed values). "
               "Edit, add or delete rows to define your own." if seed_rules else
               "No rules could be generated automatically. Add your own below.")
    rules_df = st.data_editor(
        pd.DataFrame(seed_rules, columns=["column", "op", "value"]),
        num_rows="dynamic", use_container_width=True, key=f"rules_{uploaded_file.name}",
        column_config={
            "column": st.column_config.SelectboxColumn("Column", options=list(df.columns)),
            "op": st.column_config.SelectboxColumn("Rule", options=RULE_OPS),
            "value": st.column_config.TextColumn("Value (e.g. 0 | 1,100 | A,B)"),
        })
    rule_list = [r for r in rules_df.fillna("").to_dict("records") if r["column"] and r["op"]]
    rule_results = [evaluate_rule(df, r) for r in rule_list]
    if rule_results:
        res_df = pd.DataFrame(rule_results)[["rule", "status", "failed", "pct", "note"]]
        res_df.columns = ["Rule", "Status", "Failed rows", "Failed %", "Note"]
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export rules as Great Expectations suite (.json)",
                           rules_to_great_expectations(rule_list).encode("utf-8"),
                           file_name="expectation_suite.json", mime="application/json")

    st.markdown("---")

    # ---------- run analysis ----------
    run_key = (uploaded_file.name, error_log_file.name if error_log_file else None,
               baseline_file.name if baseline_file else None)
    if st.button("🚀 Run AI Root-Cause Analysis & Copilot"):
        with st.spinner("🤖 AI Copilot is profiling data, checking rules and analyzing ETL logs..."):
            plan = build_plan(df, prof)
            st.session_state["analysis"] = {
                "key": run_key,
                "drift": detect_drift(df, baseline_df),
                "logs": analyze_logs(error_logs, df.columns),
                "plan": plan,
                "steps": plan_steps(plan, prof),
                "rules": rule_results,
                "chat": [],
            }
            st.session_state.pop("applied", None)

    an = st.session_state.get("analysis")
    if an and an["key"] == run_key:
        st.markdown(
            '<div class="report-head"><div class="ri">🧠</div><div>'
            '<div class="rt">AI Diagnostic & Remediation Report</div>'
            f'<div class="rs">Generated for {uploaded_file.name}</div></div></div>',
            unsafe_allow_html=True)

        # ---------- cleaning review: approve each fix ----------
        section("🛠️", "Review cleaning fixes before applying")
        st.caption("Tick the fixes you approve. Risky ones (merging possible label variants, capping outliers) "
                   "are off by default. Nothing is changed in your original file.")
        steps = an["steps"]
        if steps:
            steps_df = pd.DataFrame([{"Apply": s["default"], "Fix": s["fix"], "Column": s["column"],
                                      "What changes": s["detail"], "Rows": s["rows"]} for s in steps])
            edited = st.data_editor(
                steps_df, hide_index=True, use_container_width=True, key=f"steps_{an['key']}",
                disabled=["Fix", "Column", "What changes", "Rows"],
                column_config={"Apply": st.column_config.CheckboxColumn("Apply", default=False)})
            approved = [steps[i]["id"] for i, ok in enumerate(edited["Apply"].tolist()) if ok]
        else:
            st.success("No fixes are needed.")
            approved = []

        approved_key = (an["key"], tuple(approved))
        if st.session_state.get("applied", {}).get("key") != approved_key:
            applied_plan = filter_plan(an["plan"], approved)
            cleaned = apply_plan(df, applied_plan)
            prof_after = profile_dataframe(cleaned)
            st.session_state["applied"] = {
                "key": approved_key, "plan": applied_plan,
                "cleaned_csv": cleaned.to_csv(index=False).encode("utf-8"),
                "prof_after": prof_after,
                "rows_after": len(cleaned),
            }
        ap = st.session_state["applied"]
        prof_after = ap["prof_after"]

        # before -> after
        s_before, s_after = score["value"], prof_after["score"]["value"]
        m_before, m_after = prof["missing_cells"], prof_after["missing_cells"]
        h_before, h_after = hidden_cells, hidden_total(prof_after)

        def delta_txt(before, after, fmt="{:,}"):
            return "no change" if before == after else f"{fmt.format(before)} → {fmt.format(after)}"

        b1, b2, b3 = st.columns(3)
        b1.metric("Quality score", f"{s_after} / 100", delta_txt(s_before, s_after, "{:.1f}"),
                  delta_color="off" if s_before == s_after else "normal")
        b2.metric("Missing cells", f"{m_after:,}", delta_txt(m_before, m_after),
                  delta_color="off" if m_before == m_after else "inverse")
        b3.metric("Hidden / invalid cells", f"{h_after:,}", delta_txt(h_before, h_after),
                  delta_color="off" if h_before == h_after else "inverse")
        if ap["rows_after"] != prof["rows"]:
            st.caption(f"Rows: {prof['rows']:,} → {ap['rows_after']:,} (duplicates removed).")

        # ---------- report ----------
        report = build_report(prof, an["rules"], an["drift"], an["logs"], an["plan"], ap["plan"])
        st.markdown(report)

        d1, d2, d3, _ = st.columns([1, 1, 1, 1])
        d1.download_button("⬇️ Report (.md)", report.encode("utf-8"),
                           file_name="data_quality_report.md", mime="text/markdown")
        d2.download_button("⬇️ Report (.html)",
                           report_html(report, f"Data quality report - {uploaded_file.name}").encode("utf-8"),
                           file_name="data_quality_report.html", mime="text/html")
        d3.download_button("⬇️ Cleaned CSV", ap["cleaned_csv"],
                           file_name="cleaned_" + uploaded_file.name, mime="text/csv")

        # ---------- interactive copilot ----------
        section("💬", "Ask the Copilot")
        st.caption("Ask about any column, the score, the logs, or request SQL/Python fixes. "
                   "Uses Gemini (GEMINI_API_KEY in .env) or Hugging Face when a key is set; only statistics are sent, never raw rows.")
        for m in an["chat"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        problem_cols = [c for c, i in cols.items()
                        if i["missing"] or i["hidden_nulls"] or i["invalid_zeros"] or i["nonstandard_rows"]
                        or i["review_rows"] or i["scale_conflicts"]]
        c_a = problem_cols[0] if problem_cols else df.columns[0]
        c_b = problem_cols[1] if len(problem_cols) > 1 else df.columns[min(1, len(df.columns) - 1)]
        q = st.chat_input(f"e.g. Why does {c_a} have issues? How do I fix {c_b} in SQL?")
        if q:
            ctx = {"profile": {k: prof[k] for k in ("rows", "columns_count", "duplicates", "score", "columns")},
                   "rules": an["rules"], "drift": an["drift"], "logs": an["logs"], "plan": ap["plan"],
                   "steps": [{"fix": s["fix"], "column": s["column"], "what": s["detail"], "rows": s["rows"],
                              "approved": s["id"] in approved} for s in steps]}
            with st.chat_message("user"):
                st.markdown(q)
            reply, engine = ask_copilot(q, ctx, an["chat"])
            an["chat"] += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
            with st.chat_message("assistant"):
                st.markdown(reply)
                st.caption(f"answered by: {engine}")
else:
    st.info("👆 Upload a CSV dataset to begin. ETL error logs and a baseline file are optional.")