"""Interactive Q&A over the analysis. Uses Gemini (GEMINI_API_KEY), Hugging Face (HUGGINGFACEHUB_API_TOKEN)
or Claude (ANTHROPIC_API_KEY). Only statistics, category labels and log excerpts are sent, never the raw rows;
otherwise a local rule-based fallback answers from the analysis itself."""
import json
import os

try:  # your project already keeps keys in .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SYSTEM = ("You are a data-quality copilot for data engineers and business users. Answer using ONLY the "
          "JSON context (profile, rule results, drift, log findings, cleaning steps). Be concise and "
          "write in plain business language. NEVER mention JSON field names or internal keys such as "
          "'strength', 'by', 'impute', 'plan', 'label_groups' or 'KEY_FINDINGS'; say things like "
          "'Outlet_Identifier explains almost all of it' instead. Give Python or SQL only when asked. "
          "Treat root causes as likely explanations, not facts, and say when the context does not contain the answer.\n"
          "Important rules:\n"
          "1. Labels that differ only by a unit or scale word (for example 'Dollars' vs 'Dollars (millions)', "
          "'%' vs 'percent', 'thousands' vs 'millions') are DIFFERENT units, not spelling variants. Never suggest "
          "merging or replacing one with the other; suggest converting values explicitly or splitting the column, "
          "and confirming with the data owner.\n"
          "2. Possible label variants waiting for review are NOT confirmed errors; say a person must confirm them.\n"
          "3. Only describe a fix as applied if it is marked approved in the cleaning steps; otherwise call it "
          "proposed or pending.\n"
          "4. For drift, explain what moved versus the baseline (schema, numeric distribution, category shares) "
          "and give likely causes as hypotheses.")


# ----------------------------------------------------------------------
# helpers that read the analysis context
# ----------------------------------------------------------------------
def _drift_text(drift):
    if not drift or not drift.get("has_baseline"):
        return "No baseline file was uploaded, so no drift check was done."
    bits = []
    if drift.get("added"):
        bits.append(f"new columns {drift['added']}")
    if drift.get("removed"):
        bits.append(f"columns missing vs baseline {drift['removed']}")
    for d in drift.get("dtype_changed", []):
        bits.append(f"{d['column']} changed type {d['was']} -> {d['now']}")
    for d in drift.get("numeric_shift", []):
        bits.append(f"{d['column']} values shifted ({d['level']}, PSI {d['psi']})")
    for d in drift.get("category_shift", []):
        extra = f", new values {d['new_values']}" if d.get("new_values") else ""
        bits.append(f"{d['column']} category mix moved {d['tvd'] * 100:.0f}%{extra}")
    return "Compared with the baseline: " + "; ".join(bits) + "." if bits else \
        "No drift: schema and value distributions match the baseline."


def _steps_text(ctx):
    steps = ctx.get("steps") or []
    if not steps:
        return "No cleaning steps were proposed."
    ok = [s for s in steps if s.get("approved")]
    pend = [s for s in steps if not s.get("approved")]
    out = []
    if ok:
        out.append("Approved: " + "; ".join(f"{s['fix']} ({s['column']})" for s in ok) + ".")
    if pend:
        out.append("Not applied yet (needs your tick in the review table): "
                   + "; ".join(f"{s['fix']} ({s['column']})" for s in pend) + ".")
    return " ".join(out)


def _summary(ctx):
    """Small, high-signal digest placed FIRST in the prompt so truncation never drops it."""
    cols = ctx["profile"]["columns"]
    return {
        "score": ctx["profile"]["score"],
        "mixed_unit_labels_do_not_merge": {c: i["scale_conflicts"] for c, i in cols.items()
                                           if i.get("scale_conflicts")},
        "possible_label_variants_need_review": {c: i["label_review"] for c, i in cols.items()
                                                if i.get("label_review")},
        "columns_with_problems": {c: {k: i[k] for k in ("missing_pct", "hidden_nulls", "invalid_zeros", "outliers")
                                      if i.get(k)} for c, i in cols.items()
                                  if i["missing"] or i["hidden_nulls"] or i["invalid_zeros"]},
        "drift": _drift_text(ctx.get("drift")),
        "cleaning_steps": ctx.get("steps", []),
    }


def _compact(ctx, limit=8000):
    """Stats only (no raw rows), compact JSON, summary first, trimmed so small models can read it."""
    payload = {"KEY_FINDINGS": _summary(ctx), **{k: v for k, v in ctx.items() if k != "steps"}}
    return json.dumps(payload, default=str, separators=(",", ":"))[:limit]


# ----------------------------------------------------------------------
# local (no-AI) answers
# ----------------------------------------------------------------------
def _local(question, ctx):
    """Rule-based answers built from the analysis itself (works with no AI model)."""
    from src.dq_engine import plan_to_pandas, plan_to_sql
    from src.report import _root_causes, _prevention
    q = question.lower()
    prof, plan = ctx["profile"], ctx["plan"]
    cols = prof["columns"]
    hit = [c for c in cols if c.lower() in q]

    def pick(text):
        keep = [l for l in text.splitlines() if any(c in l for c in hit)] if hit else []
        return "\n".join(keep) or text

    if any(k in q for k in ("drift", "baseline", "schema")):
        return _drift_text(ctx.get("drift"))
    if any(k in q for k in ("scale", "million", "merge", "convert", "dollars")) or (hit and any(
            cols[c].get("scale_conflicts") for c in hit) and any(k in q for k in ("why", "flag", "warn"))):
        mixed = {c: i["scale_conflicts"] for c, i in cols.items() if i.get("scale_conflicts")}
        if mixed:
            return "\n".join(
                f"**{c}**: '{a}' and '{b}' differ by a unit/scale word, so they are most likely different "
                f"units, not typos. Do not merge them. Convert the values explicitly (or split the column) and "
                f"confirm with the data owner which rows use which unit."
                for c, pairs in mixed.items() for a, b in pairs[:2])
    if any(k in q for k in ("review", "approve", "pending", "apply", "applied")):
        return _steps_text(ctx)
    if "sql" in q:
        sql = pick(plan_to_sql(plan))
        note = "\n\n_No fixes are approved yet. Tick them in the review table to generate SQL._" \
            if sql.startswith("-- no approved") else ""
        return "```sql\n" + sql + "\n```" + note
    if any(k in q for k in ("python", "pandas", "code", "script")):
        return "```python\n" + pick(plan_to_pandas(plan)) + "\n```"
    if any(k in q for k in ("why", "cause", "reason", "root")):
        txt = _root_causes(prof, ctx["drift"], ctx["logs"], plan)
        lines = [l for l in txt.splitlines() if l.strip() and (not hit or any(c in l for c in hit))]
        return "\n".join(lines) or txt
    if any(k in q for k in ("prevent", "avoid", "future")):
        return _prevention(prof, plan, ctx["drift"], ctx["logs"])
    return _facts(question, ctx)


def _facts(question, ctx):
    q = question.lower()
    cols = ctx["profile"]["columns"]
    hit = [c for c in cols if c.lower() in q]
    if hit:
        out = []
        for c in hit:
            i = cols[c]
            facts = [f"{i['missing']:,} missing ({i['missing_pct']}%)"]
            if i["hidden_nulls"]:
                facts.append(f"{i['hidden_nulls']:,} placeholder values {i['hidden_tokens']}")
            if i["invalid_zeros"]:
                facts.append(f"{i['invalid_zeros']:,} zero/negative values")
            for g in i["label_groups"]:
                facts.append(f"labels {g['variants']} differ from '{g['canonical']}' only by case/spacing")
            for g in i.get("label_review", []):
                facts.append(f"labels {g['variants']} may be variants of '{g['canonical']}' (needs your review)")
            for a, b in i.get("scale_conflicts", []):
                facts.append(f"'{a}' vs '{b}' look like different units (do not merge)")
            if i["outliers"]:
                facts.append(f"{i['outliers']:,} outliers")
            step = next((s for s in ctx["plan"]["impute"] if s["col"] == c), None)
            if step and step["by"]:
                facts.append(f"can be filled from {step['by']['by']} (explains ~{step['by']['strength']:.0%})")
            out.append(f"**{c}** ({i['dtype']}, {i['unique']:,} distinct): " + "; ".join(facts) + ".")
        return "\n\n".join(out)
    s = ctx["profile"]["score"]
    if any(k in q for k in ("score", "health", "quality")):
        return (f"Quality score {s['value']}/100 ({s['label']}): average missing {s['avg_missing']}%, "
                f"duplicates {s['duplicates']}%, invalid values {s['invalid']}%, "
                f"worst-column penalty {s['column_penalty']}.")
    if "log" in q:
        lg = ctx["logs"]
        return "No logs uploaded." if not lg["provided"] else f"Log categories: {json.dumps(lg['categories'])[:1500]}"
    worst = sorted(cols.items(), key=lambda kv: -kv[1]["missing_pct"])[:3]
    return ("Local mode (set GEMINI_API_KEY or HUGGINGFACEHUB_API_TOKEN in .env for full AI answers). "
            "Worst columns by missing %: " + ", ".join(f"{c} {i['missing_pct']}%" for c, i in worst))


# ----------------------------------------------------------------------
# AI back-ends
# ----------------------------------------------------------------------
FALLBACK_MODELS = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
                   "HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen2.5-72B-Instruct"]


def _hf_answer(question, ctx, history):
    """Try HF_MODEL first, then smaller models, until one answers."""
    from huggingface_hub import InferenceClient
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    models = [m for m in [os.getenv("HF_MODEL")] + FALLBACK_MODELS if m]
    msgs = [{"role": "system", "content": SYSTEM + "\n\nContext JSON:\n" + _compact(ctx)}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    msgs.append({"role": "user", "content": question})
    errors = []
    for model in models:
        try:
            client = InferenceClient(model=model, token=token)
            r = client.chat_completion(messages=msgs, max_tokens=700, temperature=0.2)
            return r.choices[0].message.content, f"Hugging Face ({model})"
        except Exception as e:
            errors.append(f"{model.split('/')[-1]}: {str(e)[:60]}")
    raise RuntimeError(" | ".join(errors))


GEMINI_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
_SKIP = ("image", "tts", "live", "embed", "audio", "robotics", "computer", "aqa", "imagen")


def _discover_gemini_models(key):
    """Ask Google which models this key can use (names change, so don't hard-code only)."""
    import requests
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         headers={"x-goog-api-key": key}, params={"pageSize": 100}, timeout=20)
        names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception:
        return []
    names = [n for n in names if n.startswith("gemini") and not any(s in n for s in _SKIP)]
    flash = sorted((n for n in names if "flash" in n), reverse=True)  # newest-looking first
    return flash[:4] + [n for n in sorted(names, reverse=True) if n not in flash][:1]


def _gemini_answer(question, ctx, history):
    """Google Gemini via REST (needs only `requests`). Key: GEMINI_API_KEY or GOOGLE_API_KEY."""
    import requests
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    models = []
    for m in [os.getenv("GEMINI_MODEL"), GEMINI_MODELS[0]] + _discover_gemini_models(key) + GEMINI_MODELS:
        if m and m not in models:
            models.append(m)
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in history[-6:]]
    contents.append({"role": "user", "parts": [{"text": question}]})
    body = {"systemInstruction": {"parts": [{"text": SYSTEM + "\n\nContext JSON:\n" + _compact(ctx, 12000)}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900}}
    errors = []
    for model in models:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=body, timeout=40)
            if r.status_code != 200:
                try:
                    why = r.json()["error"]["message"][:70]
                except Exception:
                    why = ""
                errors.append(f"{model}: HTTP {r.status_code} {why}")
                continue
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return text, f"Gemini ({model})"
        except Exception as e:
            errors.append(f"{model}: {str(e)[:50]}")
    raise RuntimeError(" | ".join(errors))


def answer(question, ctx, history):
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        try:
            return _gemini_answer(question, ctx, history)
        except Exception as e:
            return (_local(question, ctx)
                    + f"\n\n_(Gemini unavailable: {str(e)[:160]}. Showing the built-in answer.)_"), "local"
    if os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"):
        try:
            return _hf_answer(question, ctx, history)
        except Exception as e:
            hint = ("Your Hugging Face free credits look used up (402). Check huggingface.co/settings/billing "
                    "or wait for the monthly reset. ") if "402" in str(e) else ""
            return (_local(question, ctx)
                    + f"\n\n_(AI model unavailable. {hint}Showing the built-in answer.)_"), "local"
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msgs = [{"role": m["role"], "content": m["content"]} for m in history]
            msgs.append({"role": "user", "content": question})
            r = client.messages.create(
                model=os.getenv("DQ_MODEL", "claude-sonnet-5-5"), max_tokens=900,
                system=SYSTEM + "\n\nContext JSON:\n" + _compact(ctx, 15000),
                messages=msgs)
            return r.content[0].text, "Claude"
        except Exception as e:
            return _local(question, ctx) + f"\n\n_(AI call failed: {str(e)[:120]})_", "local"
    return _local(question, ctx), "local"