import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.dq_engine import (profile_dataframe, default_rules, evaluate_rule, RULE_OPS,
                         detect_drift, analyze_logs, build_plan, apply_plan)
from src.report import build_report
from src.llm import answer as ask_copilot

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Set page config
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
    import io
    return pd.read_csv(io.BytesIO(data))


# ======================================================================
# PAGE
# ======================================================================
page_header("📋", "AI DATA QUALITY & ROOT-CAUSE COPILOT",
            "Upload your dataset and ETL error logs to generate an instant diagnostic and remediation report.")

# File uploader section with unique keys so all uploaders retain state independently
up1, up2, up3 = st.columns(3)
with up1:
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], key="main_dataset")
with up2:
    error_log_file = st.file_uploader("Upload ETL Error Logs (Optional, TXT/LOG)", type=["txt", "log"], key="error_logs")
with up3:
    baseline_file = st.file_uploader("Baseline CSV (Optional, drift check)", type=["csv"], key="baseline_csv")

# Status indicators for optional files when uploaded independently
if error_log_file is not None or baseline_file is not None:
    info_msgs = []
    if error_log_file is not None:
        info_msgs.append("📄 ETL Error Logs uploaded.")
    if baseline_file is not None:
        info_msgs.append("📊 Baseline CSV uploaded.")
    if uploaded_file is None:
        st.info("ℹ️ " + " ".join(info_msgs) + " Please upload your **Dataset (CSV)** in the first box to run the analysis.")

if uploaded_file is not None:
    df = load_csv(uploaded_file.getvalue())
    baseline_df = load_csv(baseline_file.getvalue()) if baseline_file is not None else None
    error_logs = error_log_file.getvalue().decode("utf-8", errors="replace") if error_log_file is not None else ""

    # ---------- profiling (hidden nulls, invalid zeros, label variants, outliers) ----------
    prof = profile_dataframe(df)
    score = prof["score"]
    cols = prof["columns"]
    hidden_cells = sum(i["hidden_nulls"] + i["invalid_zeros"] + i["nonstandard_rows"] + i["bad_numeric_text"]
                         for i in cols.values())

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

    # ---------- dataset preview ----------
    section("📊", "Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ---------- charts ----------
    section("📈", "Data Quality Insights")
    ch1, ch2 = st.columns([3, 2])
    n = max(prof["rows"], 1)
    rows = []
    for c, i in cols.items():
        for label, val in (("Missing", i["missing"]),
                           ("Hidden nulls", i["hidden_nulls"] + i["bad_numeric_text"]),
                           ("Invalid zeros", i["invalid_zeros"]),
                           ("Inconsistent labels", i["nonstandard_rows"])):
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
                                                   "Invalid zeros": "#f87171", "Inconsistent labels": "#60a5fa"})
            st.plotly_chart(style_fig(fig_issue), use_container_width=True)
        else:
            st.bar_chart(issue_df.pivot_table(index="Column", columns="Issue", values="Affected %", fill_value=0))

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
    st.caption("Auto-generated rules are pre-filled. Edit, add or delete rows to define your own.")
    rules_df = st.data_editor(
        pd.DataFrame(default_rules(df, prof), columns=["column", "op", "value"]),
        num_rows="dynamic", use_container_width=True, key=f"rules_{uploaded_file.name}",
        column_config={
            "column": st.column_config.SelectboxColumn("Column", options=list(df.columns)),
            "op": st.column_config.SelectboxColumn("Rule", options=RULE_OPS),
            "value": st.column_config.TextColumn("Value (e.g. 0 | 1,100 | Low Fat,Regular)"),
        })
    rule_list = [r for r in rules_df.fillna("").to_dict("records") if r["column"] and r["op"]]
    rule_results = [evaluate_rule(df, r) for r in rule_list]
    if rule_results:
        res_df = pd.DataFrame(rule_results)[["rule", "status", "failed", "pct", "note"]]
        res_df.columns = ["Rule", "Status", "Failed rows", "Failed %", "Note"]
        st.dataframe(res_df, use_container_width=True)

    st.markdown("---")

    # ---------- run analysis ----------
    run_key = (uploaded_file.name, error_log_file.name if error_log_file else None,
               baseline_file.name if baseline_file else None)
    if st.button("🚀 Run AI Root-Cause Analysis & Copilot"):
        with st.spinner("🤖 AI Copilot is profiling data, checking rules and analyzing ETL logs..."):
            drift = detect_drift(df, baseline_df)
            logs = analyze_logs(error_logs, df.columns)
            plan = build_plan(df, prof)
            cleaned = apply_plan(df, plan)
            prof_after = profile_dataframe(cleaned)
            st.session_state["result"] = {
                "key": run_key,
                "report": build_report(prof, rule_results, drift, logs, plan),
                "cleaned_csv": cleaned.to_csv(index=False).encode("utf-8"),
                "score_after": prof_after["score"],
                "missing_after": prof_after["missing_cells"],
                "hidden_after": sum(i["hidden_nulls"] + i["invalid_zeros"] + i["nonstandard_rows"]
                                    + i["bad_numeric_text"] for i in prof_after["columns"].values()),
                "chat": [],
                "ctx": {"profile": {k: prof[k] for k in ("rows", "columns_count", "duplicates", "score", "columns")},
                        "rules": rule_results, "drift": drift, "logs": logs, "plan": plan},
            }

    res = st.session_state.get("result")
    if res and res["key"] == run_key:
        st.markdown(
            '<div class="report-head"><div class="ri">🧠</div><div>'
            '<div class="rt">AI Diagnostic & Remediation Report</div>'
            f'<div class="rs">Generated for {uploaded_file.name}</div></div></div>',
            unsafe_allow_html=True)

        # before / after
        b1, b2, b3 = st.columns(3)
        b1.metric("Quality score", f"{res['score_after']['value']} / 100",
                  f"{res['score_after']['value'] - score['value']:+.1f} after cleaning")
        b2.metric("Missing cells", f"{res['missing_after']:,}",
                  f"{res['missing_after'] - prof['missing_cells']:+,} after cleaning", delta_color="inverse")
        b3.metric("Hidden / invalid cells", f"{res['hidden_after']:,}",
                  f"{res['hidden_after'] - hidden_cells:+,} after cleaning", delta_color="inverse")
        st.markdown(res["report"])

        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("⬇️ Download report (.md)", res["report"].encode("utf-8"),
                           file_name="data_quality_report.md", mime="text/markdown")
        d2.download_button("⬇️ Download cleaned CSV", res["cleaned_csv"],
                           file_name="cleaned_" + uploaded_file.name, mime="text/csv")

        # ---------- interactive copilot ----------
        section("💬", "Ask the Copilot")
        st.caption("Ask about any column, the score, the logs, or request SQL/Python fixes. "
                   "Uses Gemini (GEMINI_API_KEY in .env) or Hugging Face when a key is set; only statistics are sent, never raw rows.")
        for m in res["chat"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        q = st.chat_input("e.g. Why is Outlet_Size missing? How do I fix Item_Fat_Content in SQL?")
        if q:
            with st.chat_message("user"):
                st.markdown(q)
            reply, engine = ask_copilot(q, res["ctx"], res["chat"])
            res["chat"] += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
            with st.chat_message("assistant"):
                st.markdown(reply)
                st.caption(f"answered by: {engine}")
else:
    st.info("👆 Upload a CSV dataset to begin. ETL error logs and a baseline file are optional.")