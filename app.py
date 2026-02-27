import os
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import feedparser
import streamlit as st

# ============================================================
# AI Story Research Assistant (Research → Outline → Draft)
# - Transparency (confidence + verification guidance)
# - Human-in-the-loop (editable claims JSON)
# - Workflow orientation (not a chatbot)
# ============================================================

# Defaults can be overridden via environment variables
DEFAULT_PROVIDER = os.getenv("PROVIDER", "gemini").strip().lower()  # "openai" | "gemini"
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

# Turn ON for debugging (shows raw model outputs in sidebar)
DEBUG = os.getenv("DEBUG", "1").strip() not in {"0", "false", "False", "no", "NO"}


# -------------------------
# Prompt templates (keep these stable; minimal wrapper formatting only)
# -------------------------
PROMPT1_SYSTEM = """You are a careful research assistant. You must be transparent about uncertainty. You do NOT “verify” facts; you only assess confidence and suggest how to verify."""
PROMPT1_USER = """Topic: {topic}
Audience: {audience_or_general}
Tone: {tone}

Context headlines (may be empty):
{headlines_block}

Task:
Generate 5–7 concise factual claims or key insights about the topic that would help a writer draft a story.

For each claim provide:
- Claim (1–2 sentences max)
- Confidence (one of: Likely True | Uncertain | Needs Verification)
- Why (very brief reason for confidence, max 1 sentence)
- Suggested verification source types (2–3 items; examples: official report, peer-reviewed paper, reputable news outlet, government statistics, primary source archive)

Output MUST be valid JSON with this schema:
{{
  "claims": [
    {{
      "claim": "...",
      "confidence": "Likely True|Uncertain|Needs Verification",
      "why": "...",
      "suggested_sources": ["...", "..."]
    }}
  ]
}}"""

PROMPT2_SYSTEM = """You are an expert story editor. You create clear, usable outlines."""
PROMPT2_USER = """Topic: {topic}
Audience: {audience_or_general}
Tone: {tone}

Claims (JSON):
{claims_json}

Task:
Create a 3-act outline for a short story/article draft based on the claims.
- Act 1: Hook + framing + what’s at stake
- Act 2: Develop the core ideas, add tension/contrast, deepen context
- Act 3: Resolution + takeaway

Output MUST be valid JSON with this schema:
{{
  "outline": {{
    "act1": ["...", "..."],
    "act2": ["...", "..."],
    "act3": ["...", "..."]
  }}
}}"""

PROMPT3_SYSTEM = """You are a skilled writer. You write clean, readable prose and keep uncertainty explicit."""
PROMPT3_USER = """Topic: {topic}
Audience: {audience_or_general}
Tone: {tone}

Claims (JSON):
{claims_json}

Outline (JSON):
{outline_json}

Task:
Write a ~300-word draft based on the outline and claims.
Rules:
- Do not invent specific numbers/dates/quotes unless they are explicitly present in the claims.
- If something is uncertain, phrase it carefully and add it to verification_notes.
- Keep it readable and aligned with the chosen tone.

Output MUST be valid JSON with this schema:
{{
  "draft": "...",
  "verification_notes": ["...", "..."]
}}"""


# -------------------------
# Small helpers
# -------------------------
def audience_or_general(audience: str) -> str:
    a = (audience or "").strip()
    return a if a else "General audience"


def json_dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def safe_get_env(key: str) -> Optional[str]:
    v = os.getenv(key, "")
    v = (v or "").strip()
    return v if v else None


def current_provider() -> str:
    p = (st.session_state.get("provider") or DEFAULT_PROVIDER or "gemini").strip().lower()
    return p if p in {"openai", "gemini"} else "gemini"


def current_models() -> Tuple[str, str]:
    openai_model = (st.session_state.get("openai_model") or OPENAI_MODEL_DEFAULT).strip()
    gemini_model = (st.session_state.get("gemini_model") or GEMINI_MODEL_DEFAULT).strip()
    return openai_model or OPENAI_MODEL_DEFAULT, gemini_model or GEMINI_MODEL_DEFAULT


def provider_ready(provider: Optional[str] = None) -> Tuple[bool, str]:
    p = (provider or current_provider()).strip().lower()
    if p == "openai":
        if not safe_get_env("OPENAI_API_KEY"):
            return (
                False,
                "Missing `OPENAI_API_KEY`. Add it to your environment to use OpenAI.\n\n"
                "Example (PowerShell): `$env:OPENAI_API_KEY=\"...\"`\n"
                "Example (bash/zsh): `export OPENAI_API_KEY=\"...\"`",
            )
        return True, ""
    if p == "gemini":
        if not safe_get_env("GEMINI_API_KEY"):
            return (
                False,
                "Missing `GEMINI_API_KEY`. Add it to your environment to use Gemini.\n\n"
                "Example (PowerShell): `$env:GEMINI_API_KEY=\"...\"`\n"
                "Example (bash/zsh): `export GEMINI_API_KEY=\"...\"`",
            )
        return True, ""
    return False, f"Unknown provider '{p}'. Choose 'openai' or 'gemini'."


# -------------------------
# Headlines (RSS)
# -------------------------
def fetch_google_news_rss(topic: str, limit: int = 5, timeout_s: int = 8) -> List[Dict[str, str]]:
    """
    Fetches Google News RSS results for a query.
    Degrades gracefully: returns [] on failure (no crash).
    """
    q = (topic or "").strip()
    if not q:
        return []

    url = "https://news.google.com/rss/search"
    params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}

    try:
        resp = requests.get(url, params=params, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        items: List[Dict[str, str]] = []
        for entry in feed.entries[:limit]:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            if title and link:
                items.append({"title": title, "link": link})
        return items
    except Exception:
        return []


def headlines_block(headlines: List[Dict[str, str]]) -> str:
    if not headlines:
        return ""
    lines = []
    for i, h in enumerate(headlines, start=1):
        lines.append(f"{i}. {h['title']} — {h['link']}")
    return "\n".join(lines)


# -------------------------
# Robust JSON parsing
# -------------------------
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    t = _CODE_FENCE_RE.sub("", t).strip()
    return t


def _extract_first_json_value(text: str) -> Any:
    """
    Finds and parses the first valid JSON object/array anywhere in the text.
    Uses JSONDecoder.raw_decode to avoid greedy regex issues.
    """
    t = _strip_code_fences(text)
    if not t:
        raise ValueError("Empty model response")

    # Fast path: whole text is JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    dec = json.JSONDecoder()
    # scan for a plausible JSON start
    for i, ch in enumerate(t):
        if ch not in "{[":
            continue
        try:
            val, end = dec.raw_decode(t[i:])
            return val
        except Exception:
            continue

    # Common “member fragment” fallback:  "claims": [...]
    s = t.lstrip()
    if s.startswith('"claims"') or s.startswith('"outline"') or s.startswith('"draft"'):
        try:
            return json.loads("{\n" + s.lstrip(",") + "\n}")
        except Exception:
            pass

    raise ValueError(f"Model did not return valid JSON. First 300 chars:\n{t[:300]}")


def strict_json_loads(text: str) -> Any:
    return _extract_first_json_value(text)


# -------------------------
# Schema validation + guardrails
# -------------------------
MAX_CLAIMS = 10
MAX_CLAIM_CHARS = 300
MAX_WHY_CHARS = 200
MAX_SOURCE_ITEMS = 6
MAX_SOURCE_ITEM_CHARS = 60


def validate_claims_schema(obj: Any) -> None:
    if not isinstance(obj, dict) or "claims" not in obj or not isinstance(obj["claims"], list):
        raise ValueError("Claims JSON missing required 'claims' list.")
    if len(obj["claims"]) == 0:
        raise ValueError("Claims list is empty.")
    if len(obj["claims"]) > MAX_CLAIMS:
        raise ValueError(f"Too many claims (max {MAX_CLAIMS}).")

    for c in obj["claims"]:
        if not isinstance(c, dict):
            raise ValueError("Each claim must be an object.")
        for k in ["claim", "confidence", "why", "suggested_sources"]:
            if k not in c:
                raise ValueError(f"Claim missing '{k}'.")

        if c["confidence"] not in ["Likely True", "Uncertain", "Needs Verification"]:
            raise ValueError("Invalid confidence label in claims.")

        if not isinstance(c["suggested_sources"], list):
            raise ValueError("'suggested_sources' must be a list.")

        claim = (c.get("claim") or "").strip()
        why = (c.get("why") or "").strip()
        sources = c.get("suggested_sources") or []

        if not claim:
            raise ValueError("A claim is empty.")
        if len(claim) > MAX_CLAIM_CHARS:
            raise ValueError(f"Claim too long (max {MAX_CLAIM_CHARS} chars).")
        if len(why) > MAX_WHY_CHARS:
            raise ValueError(f"'why' too long (max {MAX_WHY_CHARS} chars).")

        if len(sources) == 0:
            raise ValueError("Each claim must include suggested_sources.")
        if len(sources) > MAX_SOURCE_ITEMS:
            raise ValueError(f"Too many suggested_sources (max {MAX_SOURCE_ITEMS}).")
        for s in sources:
            if not isinstance(s, str):
                raise ValueError("Each suggested_sources item must be a string.")
            if len(s.strip()) == 0:
                raise ValueError("A suggested_sources item is empty.")
            if len(s) > MAX_SOURCE_ITEM_CHARS:
                raise ValueError(f"A suggested_sources item is too long (max {MAX_SOURCE_ITEM_CHARS} chars).")


def validate_outline_schema(obj: Any) -> None:
    if not isinstance(obj, dict) or "outline" not in obj or not isinstance(obj["outline"], dict):
        raise ValueError("Outline JSON missing required 'outline' object.")
    for act in ["act1", "act2", "act3"]:
        if act not in obj["outline"] or not isinstance(obj["outline"][act], list):
            raise ValueError(f"Outline missing '{act}' list.")


def validate_draft_schema(obj: Any) -> None:
    if not isinstance(obj, dict) or "draft" not in obj or "verification_notes" not in obj:
        raise ValueError("Draft JSON missing required keys.")
    if not isinstance(obj["draft"], str):
        raise ValueError("'draft' must be a string.")
    if not isinstance(obj["verification_notes"], list):
        raise ValueError("'verification_notes' must be a list.")


def new_run_id() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H%M%S-%f")[:-3]


def format_run_timestamp(iso_str: str) -> str:
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return iso_str


# -------------------------
# LLM wrapper
# -------------------------
def llm_chat(messages: List[Dict[str, str]], force_json: bool = False) -> str:
    p = current_provider()
    openai_model, gemini_model = current_models()

    if p == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=safe_get_env("OPENAI_API_KEY"))

        # Prefer JSON mode if supported by the SDK/model; fall back gracefully.
        kwargs: Dict[str, Any] = dict(
            model=openai_model,
            messages=messages,
            temperature=0.2 if force_json else 0.4,
        )
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            # Older SDKs may not accept response_format.
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)

        return (resp.choices[0].message.content or "").strip()

    if p == "gemini":
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        genai.configure(api_key=safe_get_env("GEMINI_API_KEY"))
        model = genai.GenerativeModel(gemini_model)

        flattened = []
        for m in messages:
            flattened.append(f"{m['role'].upper()}:\n{m['content']}")
        prompt = "\n\n".join(flattened)

        cfg = GenerationConfig(
            temperature=0.2 if force_json else 0.4,
            response_mime_type="application/json" if force_json else None,
        )

        r = model.generate_content(prompt, generation_config=cfg)

        # Robust extraction of text for google-generativeai responses
        txt = ""
        try:
            txt = (r.text or "")
        except Exception:
            txt = ""

        if not txt:
            try:
                parts = []
                for cand in getattr(r, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", []) or []:
                        t = getattr(part, "text", "") or ""
                        if t:
                            parts.append(t)
                txt = "\n".join(parts)
            except Exception:
                txt = ""

        return (txt or "").strip()

    raise ValueError(f"Unknown provider '{p}'")


def llm_json_call(step_name: str, system_prompt: str, user_prompt: str, validator_fn) -> Dict[str, Any]:
    """
    Parse JSON strictly. If parsing/validation fails, retry once with a corrective prompt.
    Also stores raw output for debugging.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def store_debug(attempt: str, raw: str) -> None:
        if not DEBUG:
            return
        dbg = st.session_state.get("debug_raw", {}) or {}
        dbg.setdefault(step_name, {})
        dbg[step_name][attempt] = raw
        st.session_state["debug_raw"] = dbg

    # Attempt 1
    text1 = llm_chat(messages, force_json=True).strip()
    store_debug("attempt1", text1)

    try:
        obj = strict_json_loads(text1)
        validator_fn(obj)
        return obj
    except Exception as e1:
        # Retry once with useful correction signal
        corrective = (
            "You returned invalid output.\n"
            f"Error: {type(e1).__name__}: {e1}\n\n"
            "Fix the output and return ONLY valid JSON that matches the required schema. "
            "No extra text, no markdown, no code fences."
        )
        retry_messages = messages + [{"role": "user", "content": corrective}]
        text2 = llm_chat(retry_messages, force_json=True).strip()
        store_debug("attempt2", text2)

        obj2 = strict_json_loads(text2)
        validator_fn(obj2)
        return obj2


# -------------------------
# UI helpers
# -------------------------
def confidence_badge(label: str) -> str:
    if label == "Likely True":
        return "🟢 Likely True"
    if label == "Uncertain":
        return "🟡 Uncertain"
    return "🔴 Needs Verification"


def render_claim_card(i: int, c: Dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(f"**Claim {i}** — {confidence_badge(str(c.get('confidence', '')))}")
        st.write(c.get("claim", ""))
        st.caption(f"Why: {c.get('why', '')}")
        srcs = c.get("suggested_sources", []) or []
        if srcs:
            st.caption("Suggested verification source types:")
            for s in srcs:
                st.markdown(f"- {s}")


def copy_to_clipboard_button(text: str, key: str) -> None:
    # Streamlit doesn't have a guaranteed clipboard API; use a JS snippet.
    # This is best-effort and may be blocked by browser security.
    btn = st.button("Copy to clipboard", key=key)
    if btn:
        st.components.v1.html(
            f"""
<script>
navigator.clipboard.writeText({json.dumps(text)}).then(
  () => console.log("copied"),
  () => console.log("copy failed")
);
</script>
""",
            height=0,
        )
        st.success("Copied (if your browser allowed it).")


def render_run_badge() -> None:
    rid = st.session_state.get("run_id", "") or "—"
    ts = format_run_timestamp(st.session_state.get("run_started_at", "") or "")
    st.caption(f"Run ID: `{rid}` • Started: `{ts}`")


def render_pipeline_progress(active: str) -> None:
    steps = [("research", "🔍 Research"), ("outline", "🧩 Outline"), ("draft", "✍️ Draft")]
    done_research = bool(st.session_state.get("claims_json"))
    done_outline = bool(st.session_state.get("outline_json"))
    done_draft = bool(st.session_state.get("draft_json"))
    done = {"research": done_research, "outline": done_outline, "draft": done_draft}

    cols = st.columns(3)
    for idx, (k, label) in enumerate(steps):
        icon = "✅" if done.get(k) else ("🟦" if k == active else "⬜")
        cols[idx].markdown(f"{icon} **{label}**")


# -------------------------
# State management
# -------------------------
def reset_workflow(keep_config: bool = True) -> None:
    # Keep config fields optionally
    keep = {
        "provider": st.session_state.get("provider", DEFAULT_PROVIDER),
        "openai_model": st.session_state.get("openai_model", OPENAI_MODEL_DEFAULT),
        "gemini_model": st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT),
        "topic": st.session_state.get("topic", ""),
        "audience": st.session_state.get("audience", ""),
        "tone": st.session_state.get("tone", "Neutral"),
        "use_live_headlines": st.session_state.get("use_live_headlines", False),
    }

    st.session_state.headlines = []
    st.session_state.claims_json = None
    st.session_state.claims_text_edit = ""
    st.session_state.claims_confirmed_json = None
    st.session_state.outline_json = None
    st.session_state.draft_json = None
    st.session_state.last_error = ""
    st.session_state.research_done = False
    st.session_state.run_id = ""
    st.session_state.run_started_at = ""
    st.session_state.debug_last_exception = ""
    st.session_state.debug_raw = {}

    if keep_config:
        for k, v in keep.items():
            st.session_state[k] = v


def init_state() -> None:
    defaults = {
        "provider": DEFAULT_PROVIDER if DEFAULT_PROVIDER in {"openai", "gemini"} else "gemini",
        "openai_model": OPENAI_MODEL_DEFAULT,
        "gemini_model": GEMINI_MODEL_DEFAULT,
        "topic": "",
        "audience": "",
        "tone": "Neutral",
        "use_live_headlines": False,
        "headlines": [],
        "claims_json": None,
        "claims_text_edit": "",
        "claims_confirmed_json": None,
        "outline_json": None,
        "draft_json": None,
        "last_error": "",
        "research_done": False,
        "run_id": "",
        "run_started_at": "",
        "debug_last_exception": "",
        "debug_raw": {},  # {step: {attempt: raw_text}}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -------------------------
# App layout
# -------------------------
st.set_page_config(page_title="AI Story Research Assistant", page_icon="🧠", layout="wide")
init_state()

st.title("AI Story Research Assistant")
st.caption("A workflow-style prototype: Research → Outline → Draft (with transparency + human-in-the-loop editing).")

with st.expander("Design Principles", expanded=True):
    st.markdown(
        """
This prototype focuses on three core AI product principles:

**Transparency**  
AI outputs include confidence labels and verification guidance (not fact-checking).

**Human-in-the-loop**  
Users can edit and confirm claims before downstream generation.

**Workflow orientation**  
Instead of a chatbot, the system mirrors real editorial pipelines.
""".strip()
    )

ready, msg = provider_ready()
if not ready:
    st.warning(msg)

with st.sidebar:
    st.header("Configuration")

    st.session_state.provider = st.selectbox(
        "Provider",
        options=["gemini", "openai"],
        index=["gemini", "openai"].index(current_provider()),
        help="Choose which LLM provider to use. API keys must be set as env vars.",
    )

    openai_model, gemini_model = current_models()
    if current_provider() == "openai":
        st.session_state.openai_model = st.text_input("OpenAI model", value=openai_model)
    else:
        st.session_state.gemini_model = st.text_input("Gemini model", value=gemini_model)

    st.caption(
        f"Active: `{current_provider()}` • Model: `{current_models()[1] if current_provider()=='gemini' else current_models()[0]}`"
    )

    st.session_state.topic = st.text_input(
        "Topic (required)",
        value=st.session_state.topic,
        placeholder="e.g., The rise of heatwaves in Europe",
    )
    st.session_state.audience = st.text_input(
        "Audience (optional)",
        value=st.session_state.audience,
        placeholder="e.g., High school students",
    )
    st.session_state.tone = st.selectbox(
        "Tone",
        ["Journalistic", "Neutral", "Dramatic", "Comedic"],
        index=["Journalistic", "Neutral", "Dramatic", "Comedic"].index(st.session_state.tone),
    )
    st.session_state.use_live_headlines = st.toggle(
        "Use live headlines",
        value=st.session_state.use_live_headlines,
        help="Uses Google News RSS as *context only*. It does not verify claims.",
    )

    if st.session_state.use_live_headlines:
        st.info("Headlines are *context only* and may be incomplete or unrelated. Claims are not verified.")

    st.divider()
    render_run_badge()

    col_reset, col_start = st.columns([0.9, 1.1])
    with col_reset:
        if st.button("Reset workflow", use_container_width=True):
            reset_workflow(keep_config=True)
            st.success("Workflow reset.")
    with col_start:
        ready, _ = provider_ready()
        start = st.button("Start Research Workflow", type="primary", use_container_width=True, disabled=not ready)

    if DEBUG:
        with st.expander("Debug", expanded=False):
            st.caption("Raw model outputs captured per step/attempt.")
            st.json(st.session_state.get("debug_raw", {}) or {}, expanded=False)

    if start:
        # Hard reset run outputs, keep config values
        st.session_state.last_error = ""
        st.session_state.research_done = False
        st.session_state.claims_json = None
        st.session_state.claims_confirmed_json = None
        st.session_state.outline_json = None
        st.session_state.draft_json = None
        st.session_state.headlines = []
        st.session_state.claims_text_edit = ""
        st.session_state.debug_last_exception = ""
        st.session_state.debug_raw = {}

        if not st.session_state.topic.strip():
            st.session_state.last_error = "Topic is required."
        else:
            st.session_state.run_id = new_run_id()
            st.session_state.run_started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            hl: List[Dict[str, str]] = []
            if st.session_state.use_live_headlines:
                with st.spinner("Fetching headlines..."):
                    hl = fetch_google_news_rss(st.session_state.topic, limit=5)
            st.session_state.headlines = hl

            with st.spinner("Generating research claims..."):
                try:
                    p1_user = PROMPT1_USER.format(
                        topic=st.session_state.topic.strip(),
                        audience_or_general=audience_or_general(st.session_state.audience),
                        tone=st.session_state.tone,
                        headlines_block=headlines_block(st.session_state.headlines),
                    )
                    claims_obj = llm_json_call("claims", PROMPT1_SYSTEM, p1_user, validate_claims_schema)
                    st.session_state.claims_json = claims_obj
                    st.session_state.claims_text_edit = json_dumps_pretty(claims_obj)
                    st.session_state.research_done = True
                except Exception as e:
                    st.session_state.last_error = f"Research step failed: {e}"
                    if DEBUG:
                        st.session_state.debug_last_exception = f"{type(e).__name__}: {e}"

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

tabs = st.tabs(["🔍 Research & Claims", "🧩 Outline", "✍️ Draft"])


# -------------------------
# TAB 1: Research & Claims
# -------------------------
with tabs[0]:
    render_pipeline_progress("research")
    render_run_badge()

    colA, colB = st.columns([1.2, 1])

    with colA:
        st.subheader("Claims with confidence")
        st.markdown("Legend: 🟢 Likely True | 🟡 Uncertain | 🔴 Needs Verification")

        if st.session_state.use_live_headlines:
            st.markdown("**Context headlines** (RSS; context only):")
            if st.session_state.headlines:
                for h in st.session_state.headlines:
                    st.markdown(f"- [{h['title']}]({h['link']})")
            else:
                st.info("No headlines available (RSS fetch may have failed). The workflow still works.")

        if st.session_state.claims_json:
            claims = st.session_state.claims_json.get("claims", [])
            for i, c in enumerate(claims, start=1):
                render_claim_card(i, c)

            st.download_button(
                "Download claims JSON",
                data=json_dumps_pretty(st.session_state.claims_json),
                file_name="claims.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Click **Start Research Workflow** in the sidebar to generate claims.")

    with colB:
        st.subheader("Human-in-the-loop editing")
        st.caption("Edit the JSON below (rewrite claims, adjust confidence) then confirm to lock it for the next steps.")

        if st.session_state.claims_text_edit:
            edited = st.text_area(
                "Claims JSON (editable)",
                value=st.session_state.claims_text_edit,
                height=420,
                help=f"Must remain valid JSON matching the schema. Guardrails: claim ≤ {MAX_CLAIM_CHARS} chars.",
            )
            st.session_state.claims_text_edit = edited

            confirm = st.button("Confirm claims", type="primary", use_container_width=True)
            if confirm:
                try:
                    obj = strict_json_loads(edited.strip())
                    validate_claims_schema(obj)
                    st.session_state.claims_confirmed_json = obj
                    st.success("Claims confirmed and saved for Outline/Draft.")
                except Exception as e:
                    st.error(f"Could not confirm claims: {e}")
        else:
            st.warning("No claims to edit yet. Run the research step first.")


# -------------------------
# TAB 2: Outline
# -------------------------
with tabs[1]:
    render_pipeline_progress("outline")
    render_run_badge()

    st.subheader("3-act outline")

    locked_claims = st.session_state.claims_confirmed_json or st.session_state.claims_json
    if not locked_claims:
        st.info("You need claims first. Go to **🔍 Research & Claims** and confirm (or at least generate) claims.")
    else:
        col1, col2 = st.columns([0.9, 1.1])

        with col1:
            ready, _ = provider_ready()
            gen_outline = st.button("Generate Outline", type="primary", use_container_width=True, disabled=not ready)
            if gen_outline:
                with st.spinner("Generating outline..."):
                    try:
                        p2_user = PROMPT2_USER.format(
                            topic=st.session_state.topic.strip(),
                            audience_or_general=audience_or_general(st.session_state.audience),
                            tone=st.session_state.tone,
                            claims_json=json_dumps_pretty(locked_claims),
                        )
                        outline_obj = llm_json_call("outline", PROMPT2_SYSTEM, p2_user, validate_outline_schema)
                        st.session_state.outline_json = outline_obj
                        st.success("Outline generated.")
                    except Exception as e:
                        st.error(f"Outline step failed: {e}")

            st.markdown("**Inputs used**")
            st.write(
                {
                    "topic": st.session_state.topic.strip(),
                    "audience": audience_or_general(st.session_state.audience),
                    "tone": st.session_state.tone,
                    "claims_source": "confirmed" if st.session_state.claims_confirmed_json else "generated",
                }
            )

        with col2:
            if st.session_state.outline_json:
                o = st.session_state.outline_json["outline"]
                st.markdown("### Act 1 — Hook")
                st.write(o.get("act1", []))
                st.markdown("### Act 2 — Development / Conflict")
                st.write(o.get("act2", []))
                st.markdown("### Act 3 — Resolution")
                st.write(o.get("act3", []))

                st.download_button(
                    "Download outline JSON",
                    data=json_dumps_pretty(st.session_state.outline_json),
                    file_name="outline.json",
                    mime="application/json",
                    use_container_width=True,
                )

                with st.expander("Show outline JSON"):
                    st.code(json_dumps_pretty(st.session_state.outline_json), language="json")
            else:
                st.info("Click **Generate Outline** to create a 3-act structure.")


# -------------------------
# TAB 3: Draft
# -------------------------
with tabs[2]:
    render_pipeline_progress("draft")
    render_run_badge()

    st.subheader("Draft (~300 words) + Verification Notes")

    locked_claims = st.session_state.claims_confirmed_json or st.session_state.claims_json
    outline_obj = st.session_state.outline_json

    if not locked_claims:
        st.info("You need claims first (Tab 1).")
    elif not outline_obj:
        st.info("You need an outline first (Tab 2).")
    else:
        col1, col2 = st.columns([0.9, 1.1])

        with col1:
            ready, _ = provider_ready()
            gen_draft = st.button("Generate Draft", type="primary", use_container_width=True, disabled=not ready)
            if gen_draft:
                with st.spinner("Generating draft..."):
                    try:
                        p3_user = PROMPT3_USER.format(
                            topic=st.session_state.topic.strip(),
                            audience_or_general=audience_or_general(st.session_state.audience),
                            tone=st.session_state.tone,
                            claims_json=json_dumps_pretty(locked_claims),
                            outline_json=json_dumps_pretty(outline_obj),
                        )
                        draft_obj = llm_json_call("draft", PROMPT3_SYSTEM, p3_user, validate_draft_schema)
                        st.session_state.draft_json = draft_obj
                        st.success("Draft generated.")
                    except Exception as e:
                        st.error(f"Draft step failed: {e}")

            st.markdown("**Tip**: If the copy button fails, click inside the draft text area and use **Ctrl/Cmd + C**.")

        with col2:
            if st.session_state.draft_json:
                draft_text = st.session_state.draft_json.get("draft", "")
                notes = st.session_state.draft_json.get("verification_notes", [])

                copy_to_clipboard_button(draft_text, key="copy-draft-btn")

                st.text_area("Draft (editable copy buffer)", value=draft_text, height=280)
                st.markdown("#### Verification Notes")
                if notes:
                    for n in notes:
                        st.markdown(f"- {n}")
                else:
                    st.caption("No uncertain claims listed (or the model returned none).")

                st.download_button(
                    "Download draft JSON",
                    data=json_dumps_pretty(st.session_state.draft_json),
                    file_name="draft.json",
                    mime="application/json",
                    use_container_width=True,
                )

                with st.expander("Show full draft JSON"):
                    st.code(json_dumps_pretty(st.session_state.draft_json), language="json")
            else:
                st.info("Click **Generate Draft** to produce the final output.")
