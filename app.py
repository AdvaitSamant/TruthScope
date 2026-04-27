import streamlit as st
import requests
import urllib.parse
import base64
import io
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.enums import TA_CENTER

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthScope",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "TruthScope: Verify facts with global databases and Vera AI reasoning"}
)

# AI persona name and model constants
AI_NAME    = "Vera"
LLM_MODEL  = "openai/gpt-oss-120b"
LLM_BASE   = "https://openrouter.ai/api/v1/chat/completions"

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in {
    'history':        [],
    'results':        None,
    'llm_analysis':   None,
    'last_query':     "",
    'page':           "landing",
    'language':       "en",
    'show_deep_dive': False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIMILARITY MODEL ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sim_model = load_model()

# ── LANGUAGE ──────────────────────────────────────────────────────────────────
LANG_MAP = {
    "English": "en", "Hindi": "hi", "Marathi": "mr",
    "French": "fr",  "German": "de", "Spanish": "es",
    "Chinese (Simplified)": "zh-CN", "Japanese": "ja"
}

@st.cache_data(ttl=3600, show_spinner=False)
def tr(text: str, lang: str) -> str:
    if not text or lang == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text) or text
    except Exception:
        return text

def T(text, lang=None):
    return tr(text, lang or st.session_state.language)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@400;500&display=swap');

:root {
  --b50:#fdf8f3; --b100:#f5ece0; --b200:#e8d5bc; --b300:#d4b896;
  --b400:#b8926a; --b500:#9b7145; --b600:#7d5632; --b700:#5c3d1e; --b800:#3d2710;
  --teal:#2a9d8f; --teal-lt:#d4f1ee;
  --amber:#e9a825; --amber-lt:#fef3d0;
  --red:#c0392b;   --red-lt:#fde8e8;
  --green:#27ae60; --green-lt:#d5f5e3;
  --tp:#2c1a0e; --ts:#6b4c33; --tm:#a07850;
  --bg:#fdf8f3; --card:#fffcf8; --border:#e8d5bc;
}

*{margin:0;padding:0;box-sizing:border-box;}

html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important;
  font-family:'Source Serif 4',Georgia,serif;
  color:var(--tp);
}

.block-container{
  padding-top:2.8rem!important;
  padding-bottom:2.5rem!important;
  max-width:1100px!important;
}

[data-testid="stHeader"]{background:var(--bg)!important;border-bottom:1px solid var(--border);}
[data-testid="stSidebar"]{background:var(--b100)!important;}

/* ── NAVBAR ROW  ── */
/* All navbar elements share the same row height via flex alignment */
.nav-cell{
  display:flex;
  align-items:center;
  height:52px;           /* fixed row height — every cell matches */
}
.nav-cell-center{ justify-content:center; }
.nav-cell-right { justify-content:flex-end; gap:8px; }
.nav-cell-left  { justify-content:flex-start; }

.nav-logo{
  font-family:'Playfair Display',serif;
  font-size:1.6rem;font-weight:800;
  color:var(--b700);letter-spacing:-.5px;line-height:1;
}
.nav-logo span{color:var(--teal);}
.nav-tagline{font-size:.74rem;color:var(--tm);font-style:italic;margin-top:2px;}
.nav-logo-wrap{display:flex;flex-direction:column;align-items:center;}

.lang-lbl{
  font-family:'DM Mono',monospace;font-size:.62rem;font-weight:500;
  letter-spacing:1.6px;text-transform:uppercase;color:var(--tm);
  white-space:nowrap;line-height:1;
}

/* Make the selectbox fill its column and align vertically */
[data-testid="stSelectbox"]{margin-top:0!important;}
[data-testid="stSelectbox"] label{display:none!important;}
[data-testid="stSelectbox"]>div>div{
  border:1.5px solid var(--b300)!important;
  border-radius:8px!important;
  background:var(--b50)!important;
  font-family:'Source Serif 4',serif!important;
  font-size:.84rem!important;
  color:var(--tp)!important;
  min-height:36px!important;
}

/* ── TYPE SCALE ── */
.t-hero{
  font-family:'Playfair Display',serif;
  font-size:clamp(2.5rem,5vw,4rem);font-weight:800;
  color:var(--b800);letter-spacing:-1.5px;line-height:1.06;
}
.t-section{
  font-family:'Playfair Display',serif;
  font-size:clamp(1.3rem,2.2vw,1.8rem);font-weight:700;
  color:var(--b700);letter-spacing:-.4px;line-height:1.25;
}
.t-caption{font-size:.82rem;color:var(--tm);font-style:italic;}
.section-label{
  font-family:'DM Mono',monospace;font-size:.62rem;font-weight:500;
  letter-spacing:2px;text-transform:uppercase;color:var(--b400);margin-bottom:5px;
}

/* ── DIVIDER ── */
.divider{
  height:1px;
  background:linear-gradient(90deg,transparent,var(--b300),transparent);
  margin:22px 0;
}

/* ── HERO ── */
.hero-wrap{
  background:linear-gradient(135deg,var(--b100) 0%,var(--b50) 55%,#fff9f2 100%);
  border:1px solid var(--border);border-radius:18px;
  padding:50px 46px 42px;margin-bottom:10px;
  position:relative;overflow:hidden;
}
.hero-wrap::before{
  content:'';position:absolute;top:-40px;right:-40px;
  width:230px;height:230px;
  background:radial-gradient(circle,var(--b200) 0%,transparent 70%);
  opacity:.45;border-radius:50%;
}
.hero-wrap::after{
  content:'"';position:absolute;bottom:-28px;right:26px;
  font-family:'Playfair Display',serif;
  font-size:15rem;color:var(--b200);opacity:.32;line-height:1;
}
.hero-pill{
  display:inline-block;
  background:var(--amber-lt);color:var(--b600);
  border:1px solid var(--amber);border-radius:100px;
  padding:3px 12px;font-family:'DM Mono',monospace;
  font-size:.66rem;font-weight:500;letter-spacing:1.4px;
  text-transform:uppercase;margin-bottom:14px;
}
.hero-sub{
  font-size:1.05rem;color:var(--ts);
  max-width:490px;line-height:1.75;margin-top:10px;font-weight:300;
}

/* ── FEATURE GRID ── */
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin:14px 0 4px;}
.feat-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:11px;padding:18px 16px;
  transition:box-shadow .2s,transform .2s;
}
.feat-card:hover{box-shadow:0 5px 16px rgba(92,61,30,.1);transform:translateY(-2px);}
.feat-icon{font-size:1.3rem;margin-bottom:7px;display:block;color:var(--b500);}
.feat-title{font-size:.9rem;font-weight:600;color:var(--tp);margin-bottom:4px;}
.feat-desc{font-size:.78rem;color:var(--tm);line-height:1.5;}

/* ── AI ENGINE CARD ── */
.ai-card{
  background:linear-gradient(135deg,var(--b700) 0%,var(--b600) 100%);
  border-radius:14px;padding:0;margin:14px 0 4px;
  color:white;overflow:hidden;
  display:grid;grid-template-columns:1fr 1fr;
}
.ai-card-left{padding:26px 28px;}
.ai-card-right{
  padding:26px 28px;
  border-left:1px solid rgba(255,255,255,.1);
  background:rgba(0,0,0,.08);
}
.ai-card-eyebrow{
  font-family:'DM Mono',monospace;font-size:.6rem;font-weight:500;
  letter-spacing:2px;text-transform:uppercase;
  color:rgba(255,255,255,.45);margin-bottom:6px;
}
.ai-card-name{
  font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:800;
  color:white;margin-bottom:4px;letter-spacing:-.3px;line-height:1.1;
}
.ai-card-model{
  font-family:'DM Mono',monospace;font-size:.65rem;
  color:rgba(255,255,255,.5);letter-spacing:.6px;margin-bottom:12px;
}
.ai-card-body{font-size:.84rem;color:rgba(255,255,255,.82);line-height:1.65;}
.ai-card-tag{
  display:inline-block;background:rgba(255,255,255,.12);
  border:1px solid rgba(255,255,255,.2);border-radius:4px;
  padding:2px 8px;font-family:'DM Mono',monospace;
  font-size:.64rem;color:rgba(255,255,255,.8);margin:3px 2px 0;
}
.ai-card-right h5{
  font-family:'Playfair Display',serif;font-size:.9rem;font-weight:700;
  color:rgba(255,255,255,.9);margin-bottom:8px;margin-top:0;
}
.ai-card-right p{font-size:.8rem;color:rgba(255,255,255,.7);line-height:1.6;margin-bottom:10px;}

/* ── STEPS ROW ── */
.steps-row{
  display:grid;grid-template-columns:repeat(5,1fr);
  background:var(--b100);border:1px solid var(--border);
  border-radius:11px;overflow:hidden;margin:8px 0 16px;
}
.step-item{padding:18px 12px;text-align:center;border-right:1px solid var(--border);}
.step-item:last-child{border-right:none;}
.step-num{font-family:'Playfair Display',serif;font-size:1.7rem;font-weight:800;color:var(--b300);line-height:1;margin-bottom:5px;}
.step-text{font-size:.75rem;color:var(--ts);line-height:1.4;}

/* ── DEEP DIVE PANEL ── */
.deep-panel{
  background:var(--card);border:1px solid var(--border);
  border-radius:13px;padding:26px 28px;margin:10px 0 18px;
}
.deep-panel h4{
  font-family:'Playfair Display',serif;
  font-size:1rem;font-weight:700;color:var(--b700);
  margin-bottom:5px;margin-top:16px;
}
.deep-panel h4:first-child{margin-top:0;}
.deep-panel p{font-size:.86rem;color:var(--ts);line-height:1.65;margin-bottom:3px;}
.mono-tag{
  display:inline-block;background:var(--b100);color:var(--b600);
  border:1px solid var(--b200);border-radius:4px;
  padding:1px 6px;font-family:'DM Mono',monospace;
  font-size:.7rem;margin:0 2px 3px;
}

/* ── STREAMLIT BUTTON OVERRIDES ── */
[data-testid="baseButton-primary"]{
  background:linear-gradient(135deg,var(--b600) 0%,var(--b500) 100%)!important;
  color:white!important;
  font-family:'Source Serif 4',serif!important;
  font-size:.9rem!important;font-weight:600!important;
  border-radius:9px!important;border:none!important;
  box-shadow:0 2px 8px rgba(92,61,30,.2)!important;
  transition:all .2s!important;
}
[data-testid="baseButton-primary"]:hover{
  background:linear-gradient(135deg,var(--b700) 0%,var(--b600) 100%)!important;
  box-shadow:0 5px 14px rgba(92,61,30,.3)!important;
  transform:translateY(-1px)!important;
}
[data-testid="baseButton-secondary"]{
  background:var(--b100)!important;color:var(--b700)!important;
  border:1.5px solid var(--b300)!important;
  font-family:'Source Serif 4',serif!important;
  font-size:.86rem!important;border-radius:9px!important;
}
[data-testid="baseButton-secondary"]:hover{
  background:var(--b200)!important;
  box-shadow:0 3px 10px rgba(92,61,30,.12)!important;
}

/* ── INPUTS ── */
textarea,input[type="text"]{
  background:var(--card)!important;
  border:1.5px solid var(--border)!important;
  border-radius:9px!important;
  font-family:'Source Serif 4',serif!important;
  font-size:.93rem!important;color:var(--tp)!important;
}
textarea:focus,input[type="text"]:focus{
  border-color:var(--b400)!important;
  box-shadow:0 0 0 3px rgba(155,113,69,.12)!important;
}

/* ── RESULT CARDS ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
.result-card{
  background:var(--card);border:1px solid var(--border);
  border-left:4px solid var(--b400);border-radius:11px;
  padding:19px 22px 15px;margin-bottom:11px;
  animation:fadeUp .35s ease-out both;
  transition:box-shadow .2s;
}
.result-card:hover{box-shadow:0 5px 16px rgba(92,61,30,.09);}

/* ── VERA ANALYSIS CARD ── */
.vera-card{
  background:linear-gradient(135deg,var(--b800) 0%,var(--b700) 100%);
  border-radius:13px;padding:24px 26px;margin:18px 0 6px;
  position:relative;overflow:hidden;
}
.vera-card::before{
  content:'';position:absolute;top:-30px;right:-30px;
  width:160px;height:160px;
  background:radial-gradient(circle,rgba(255,255,255,.06) 0%,transparent 70%);
  border-radius:50%;
}
.vera-header{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.vera-badge{
  font-family:'DM Mono',monospace;font-size:.6rem;font-weight:500;
  letter-spacing:1.8px;text-transform:uppercase;
  background:rgba(255,255,255,.12);color:rgba(255,255,255,.75);
  border:1px solid rgba(255,255,255,.18);border-radius:100px;
  padding:2px 10px;
}
.vera-model{
  font-family:'DM Mono',monospace;font-size:.58rem;
  color:rgba(255,255,255,.35);letter-spacing:.6px;
}
.vera-verdict{
  font-size:.97rem;font-weight:600;color:white;
  line-height:1.5;margin-bottom:12px;
  font-family:'Source Serif 4',serif;
}
.vera-body{
  font-size:.88rem;color:rgba(255,255,255,.88);
  line-height:1.75;font-weight:300;
  font-family:'Source Serif 4',serif;
}

/* ── NO RESULTS CARD ── */
.no-res-card{
  background:var(--b100);border:1px dashed var(--b300);
  border-radius:11px;padding:26px;text-align:center;margin:6px 0;
}
.no-res-title{
  font-family:'Playfair Display',serif;
  font-size:1.05rem;font-weight:700;color:var(--b700);margin-bottom:7px;
}
.no-res-body{font-size:.85rem;color:var(--ts);line-height:1.65;max-width:460px;margin:0 auto;}

/* ── RESULT META ── */
.match-badge{
  display:inline-flex;align-items:center;gap:4px;
  padding:2px 9px;border-radius:100px;
  font-family:'DM Mono',monospace;font-size:.63rem;
  font-weight:500;letter-spacing:.8px;text-transform:uppercase;
  color:white;margin-bottom:10px;
}
.claim-text{font-size:.98rem;font-weight:600;color:var(--tp);line-height:1.55;margin-bottom:10px;}
.meta-row{display:flex;align-items:flex-start;gap:8px;font-size:.83rem;color:var(--ts);margin-bottom:4px;}
.meta-lbl{
  font-family:'DM Mono',monospace;font-size:.63rem;
  font-weight:500;letter-spacing:.8px;text-transform:uppercase;
  color:var(--tm);min-width:80px;padding-top:1px;
}
.pill{display:inline-block;padding:2px 9px;border-radius:100px;font-family:'DM Mono',monospace;font-size:.66rem;font-weight:500;}
.pill-false {background:var(--red-lt);  color:var(--red);}
.pill-true  {background:var(--green-lt);color:var(--green);}
.pill-mixed {background:var(--amber-lt);color:var(--b600);}
.pill-unrated{background:var(--teal-lt);color:var(--teal);}

/* ── HISTORY PANEL ── */
/* Sidebar controls: all same width, stacked cleanly */
.sidebar-ctrl{width:100%;margin-bottom:8px;}

.hist-item{
  background:var(--b100);border:1px solid var(--border);
  border-radius:7px;padding:8px 10px;margin-bottom:5px;
  font-size:.78rem;color:var(--ts);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}

/* ── PDF DOWNLOAD BUTTON ── */
/* Rendered as a full-width anchor button to match Streamlit button sizing */
.pdf-btn{display:block;width:100%;margin-bottom:8px;}
.pdf-btn a{display:block;width:100%;text-decoration:none;}
.pdf-btn a button{
  width:100%;
  background:linear-gradient(135deg,var(--b600),var(--b500));
  color:white;border:none;
  padding:8px 16px;border-radius:9px;
  font-family:'Source Serif 4',serif;
  font-size:.86rem;font-weight:600;
  cursor:pointer;
  box-shadow:0 2px 8px rgba(92,61,30,.2);
  transition:all .2s;
  text-align:center;
}
.pdf-btn a button:hover{
  background:linear-gradient(135deg,var(--b700),var(--b600));
  box-shadow:0 4px 12px rgba(92,61,30,.3);
  transform:translateY(-1px);
}

/* Inline (results area) pdf button — auto width */
.pdf-btn-inline{display:inline-block;margin-bottom:12px;}
.pdf-btn-inline a{text-decoration:none;}
.pdf-btn-inline a button{
  background:linear-gradient(135deg,var(--b600),var(--b500));
  color:white;border:none;
  padding:8px 20px;border-radius:9px;
  font-family:'Source Serif 4',serif;
  font-size:.86rem;font-weight:600;
  cursor:pointer;
  box-shadow:0 2px 8px rgba(92,61,30,.2);
  transition:all .2s;
}
.pdf-btn-inline a button:hover{
  background:linear-gradient(135deg,var(--b700),var(--b600));
  box-shadow:0 4px 12px rgba(92,61,30,.3);
  transform:translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def rating_pill(rating):
    r = rating.lower()
    if any(w in r for w in ['false','fake','pants on fire','incorrect','wrong']): return "pill-false"
    if any(w in r for w in ['true','correct','accurate','confirmed']):            return "pill-true"
    if any(w in r for w in ['mixture','misleading','half','partly','mostly']):    return "pill-mixed"
    return "pill-unrated"

def sim_color(score):
    if score >= 75: return "#27ae60"
    if score >= 50: return "#e9a825"
    return "#c0392b"

def fact_check(query, api_key):
    url = (
        "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        f"?query={urllib.parse.quote(query)}&languageCode=en&key={api_key}"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        if not data or 'claims' not in data:
            return "no_results"
        out = []
        for claim in data['claims']:
            if 'claimReview' in claim and claim['claimReview']:
                rev = claim['claimReview'][0]
                out.append({
                    "claim":        claim.get('text',''),
                    "made_by":      claim.get('claimant','Unknown'),
                    "fact_checker": rev.get('publisher',{}).get('name','Unknown'),
                    "rating":       rev.get('textualRating','Unrated'),
                    "source_link":  rev.get('url','#'),
                })
        return out if out else "no_results"
    except requests.exceptions.RequestException as e:
        return f"error:{str(e)[:60]}"

def add_scores(results, query_text):
    q = sim_model.encode(query_text, convert_to_tensor=True)
    for res in results:
        c = sim_model.encode(res['claim'], convert_to_tensor=True)
        res['similarity_score'] = max(0, min(100, int(util.cos_sim(q, c).item() * 100)))
    return sorted(results, key=lambda x: x['similarity_score'], reverse=True)


# ── VERA AI (GPT-OSS-120B via OpenRouter) ────────────────────────────────────

def build_prompt(user_query, results):
    if not results:
        ctx = "No matching records were found in the fact-check database for this claim."
    else:
        ctx = "\n".join(
            f"Result {i}: Claim: {r['claim']} | Claimant: {r['made_by']} | "
            f"Checked by: {r['fact_checker']} | Rating: {r['rating']} | "
            f"Match score: {r['similarity_score']}%"
            for i, r in enumerate(results, 1)
        )
    return (
        f"You are {AI_NAME}, TruthScope's expert AI fact-checking analyst. "
        f"Speak in first person as {AI_NAME} throughout your response. "
        f"A user has submitted the following claim for verification:\n\n"
        f"USER CLAIM: {user_query}\n\n"
        f"FACT-CHECK DATABASE RESULTS:\n{ctx}\n\n"
        f"Answering the user's query is the first priority and should be ansered first using the evidence. "
        f"Provide a clear contextual analysis. Explain what the evidence suggests, "
        f"reference the most relevant results, discuss the credibility of the sources, "
        f"and state your overall assessment. "
        f"Write in plain, authoritative prose without bullet points or any dashes. "
        f"Do not use markdown headers."
    )

def call_vera(user_query, results, llm_key):
    """Two-turn reasoning chain. Returns (analysis_str, verdict_str)."""
    prompt = build_prompt(user_query, results)
    headers = {"Authorization": f"Bearer {llm_key}", "Content-Type": "application/json"}

    payload1 = {
        "model":    LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 900,
    }
    try:
        r1 = requests.post(LLM_BASE, headers=headers,
                           data=json.dumps(payload1), timeout=50)
        r1.raise_for_status()
        msg1 = r1.json()['choices'][0]['message']
        analysis = (msg1.get('content') or "").strip()

        # Turn 2: ask Vera to distil a single verdict sentence
        payload2 = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": analysis,
                 "reasoning_details": msg1.get('reasoning_details')},
                {"role": "user",      "content": (
                    f"As {AI_NAME}, state your final one-sentence verdict on whether "
                    f"this claim is true, false, misleading, or unverified, and why. "
                    f"Be direct and concise. Do not use dashes."
                )},
            ],
            "max_tokens": 160,
        }
        r2 = requests.post(LLM_BASE, headers=headers,
                           data=json.dumps(payload2), timeout=35)
        verdict = ""
        if r2.status_code == 200:
            verdict = (r2.json()['choices'][0]['message'].get('content') or "").strip()

        return analysis, verdict
    except Exception as e:
        return None, f"{AI_NAME} is unavailable: {str(e)[:80]}"


# ── PDF BUILDERS ──────────────────────────────────────────────────────────────

def _styles():
    br    = colors.HexColor("#5c3d1e")
    muted = colors.HexColor("#a07850")
    sec   = colors.HexColor("#6b4c33")
    light = colors.HexColor("#f5ece0")
    return br, muted, sec, light

def build_single_pdf(query, results, analysis=None, verdict=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
          leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    br, muted, sec, light = _styles()

    H  = ParagraphStyle('H',  fontName='Times-Bold',   fontSize=22, textColor=br,   spaceAfter=3,  leading=26)
    Su = ParagraphStyle('Su', fontName='Times-Italic', fontSize=10, textColor=muted, spaceAfter=14, leading=14)
    Lb = ParagraphStyle('Lb', fontName='Courier-Bold', fontSize=7.5,textColor=muted, spaceAfter=1,  leading=10)
    Cl = ParagraphStyle('Cl', fontName='Times-Bold',   fontSize=11, textColor=colors.HexColor("#2c1a0e"), spaceAfter=8, leading=16)
    Bo = ParagraphStyle('Bo', fontName='Times-Roman',  fontSize=10, textColor=sec,   spaceAfter=5,  leading=15)
    Fo = ParagraphStyle('Fo', fontName='Times-Italic', fontSize=8,  textColor=muted, alignment=TA_CENTER)

    story = [
        Paragraph("TruthScope", H),
        Paragraph(f"Fact Check Report   {datetime.now().strftime('%d %B %Y, %H:%M')}", Su),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e8d5bc"), spaceAfter=10),
        Paragraph("QUERY", Lb),
        Paragraph(query, Cl),
        HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#e8d5bc"), spaceAfter=12),
    ]

    if verdict:
        story += [Paragraph(f"{AI_NAME.upper()} VERDICT", Lb), Paragraph(verdict, Bo), Spacer(1,.15*cm)]
    if analysis:
        story += [Paragraph(f"{AI_NAME.upper()} ANALYSIS", Lb),
                  Paragraph(analysis.replace('\n','<br/>'), Bo),
                  Spacer(1,.2*cm),
                  HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#e8d5bc"), spaceAfter=12)]

    for i, res in enumerate(results, 1):
        story.append(Paragraph(f"DATABASE RESULT {i}", Lb))
        story.append(Paragraph(res['claim'], Cl))
        tbl = Table([
            ["Match",       f"{res['similarity_score']}%"],
            ["Claimant",    res['made_by']],
            ["Verified by", res['fact_checker']],
            ["Rating",      res['rating']],
            ["Source",      res['source_link']],
        ], colWidths=[3.2*cm, 13.3*cm])
        tbl.setStyle(TableStyle([
            ('FONTNAME',  (0,0),(0,-1),'Courier-Bold'), ('FONTSIZE',(0,0),(0,-1),7),
            ('TEXTCOLOR', (0,0),(0,-1), muted),
            ('FONTNAME',  (1,0),(1,-1),'Times-Roman'),  ('FONTSIZE',(1,0),(1,-1),9.5),
            ('TEXTCOLOR', (1,0),(1,-1), sec),
            ('VALIGN',    (0,0),(-1,-1),'TOP'),
            ('ROWBACKGROUNDS',(0,0),(-1,-1),[light, colors.white]),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('LEFTPADDING',(0,0),(-1,-1),6),
        ]))
        story += [tbl, Spacer(1,.4*cm)]
        if i < len(results):
            story.append(HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#e8d5bc"), spaceAfter=10))

    story += [Spacer(1,.7*cm),
              HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#e8d5bc"), spaceAfter=5),
              Paragraph("Generated by TruthScope", Fo)]
    doc.build(story); buf.seek(0); return buf.read()


def build_session_pdf(history):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
          leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    br, muted, sec, light = _styles()

    H  = ParagraphStyle('H',  fontName='Times-Bold',   fontSize=22, textColor=br,   spaceAfter=3,  leading=26)
    Su = ParagraphStyle('Su', fontName='Times-Italic', fontSize=10, textColor=muted, spaceAfter=14, leading=14)
    Lb = ParagraphStyle('Lb', fontName='Courier-Bold', fontSize=7,  textColor=muted, spaceAfter=1,  leading=10)
    Q  = ParagraphStyle('Q',  fontName='Times-Bold',   fontSize=11, textColor=colors.HexColor("#2c1a0e"), spaceAfter=6, leading=16)
    Va = ParagraphStyle('Va', fontName='Times-Roman',  fontSize=9,  textColor=sec,   spaceAfter=4,  leading=13)
    Ai = ParagraphStyle('Ai', fontName='Times-Italic', fontSize=9,  textColor=sec,   spaceAfter=5,  leading=14)
    Fo = ParagraphStyle('Fo', fontName='Times-Italic', fontSize=8,  textColor=muted, alignment=TA_CENTER)

    story = [
        Paragraph("TruthScope", H),
        Paragraph(f"Session Report   {datetime.now().strftime('%d %B %Y, %H:%M')}   {len(history)} check(s)", Su),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e8d5bc"), spaceAfter=14),
    ]

    for ci, chk in enumerate(history, 1):
        story += [Paragraph(f"CHECK {ci}", Lb), Paragraph(chk['query'], Q)]
        if chk.get('llm_verdict'):
            story += [Paragraph(f"{AI_NAME.upper()} VERDICT", Lb), Paragraph(chk['llm_verdict'], Ai)]
        if chk.get('llm_analysis'):
            story += [Paragraph(f"{AI_NAME.upper()} ANALYSIS", Lb),
                      Paragraph(chk['llm_analysis'].replace('\n','<br/>'), Ai), Spacer(1,.12*cm)]
        if not chk['results']:
            story.append(Paragraph("No verified records found.", Va))
        else:
            for i, res in enumerate(chk['results'], 1):
                tbl = Table([
                    ["#", str(i)], ["Match", f"{res['similarity_score']}%"],
                    ["Claim", res['claim']], ["Claimant", res['made_by']],
                    ["Checker", res['fact_checker']], ["Rating", res['rating']],
                    ["Source", res['source_link']],
                ], colWidths=[2.5*cm, 14*cm])
                tbl.setStyle(TableStyle([
                    ('FONTNAME',(0,0),(0,-1),'Courier-Bold'),('FONTSIZE',(0,0),(0,-1),7),
                    ('TEXTCOLOR',(0,0),(0,-1),muted),
                    ('FONTNAME',(1,0),(1,-1),'Times-Roman'),('FONTSIZE',(1,0),(1,-1),9),
                    ('TEXTCOLOR',(1,0),(1,-1),sec),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),
                    ('ROWBACKGROUNDS',(0,0),(-1,-1),[light,colors.white]),
                    ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
                    ('LEFTPADDING',(0,0),(-1,-1),5),
                ]))
                story += [tbl, Spacer(1,.18*cm)]
        story.append(Spacer(1,.3*cm))
        if ci < len(history):
            story.append(HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#e8d5bc"), spaceAfter=8))

    story += [Spacer(1,.7*cm),
              HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#e8d5bc"), spaceAfter=5),
              Paragraph("Generated by TruthScope", Fo)]
    doc.build(story); buf.seek(0); return buf.read()


def pdf_btn_sidebar(label, pdf_bytes, filename):
    """Full-width PDF download button for the sidebar."""
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(
        f'<div class="pdf-btn"><a href="data:application/pdf;base64,{b64}" '
        f'download="{filename}"><button>{label}</button></a></div>',
        unsafe_allow_html=True
    )

def pdf_btn_inline(label, pdf_bytes, filename):
    """Auto-width PDF download button for the results area."""
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(
        f'<div class="pdf-btn-inline"><a href="data:application/pdf;base64,{b64}" '
        f'download="{filename}"><button>{label}</button></a></div>',
        unsafe_allow_html=True
    )


# ── NAVBAR HELPER ─────────────────────────────────────────────────────────────
# Renders the right-side language control the same way on both pages.
# `col_lbl` and `col_sel` are already-created st.columns objects.
def render_lang_control(col_lbl, col_sel, key, lang):
    with col_lbl:
        # Wrap in nav-cell so it aligns to the same 52 px row
        st.markdown("<div class='nav-cell nav-cell-right'><span class='lang-lbl'>Language</span></div>",
                    unsafe_allow_html=True)
    with col_sel:
        st.markdown("<div style='padding-top:8px;'>", unsafe_allow_html=True)
        sel = st.selectbox(
            "Language", list(LANG_MAP.keys()),
            index=list(LANG_MAP.values()).index(lang),
            key=key, label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return LANG_MAP[sel]


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def landing(lang):

    # Navbar: logo left | spacer | LANGUAGE label | selectbox
    # All four columns sit inside a shared 52 px tall row via .nav-cell
    c_logo, c_mid, c_lbl, c_sel = st.columns([3, 3, 1, 1.4])

    with c_logo:
        st.markdown("<div class='nav-cell nav-cell-left'>"
                    "<div class='nav-logo'>Truth<span>Scope</span></div>"
                    "</div>", unsafe_allow_html=True)

    with c_lbl:
        st.markdown("<div class='nav-cell nav-cell-right'>"
                    "<span class='lang-lbl'>Language</span></div>",
                    unsafe_allow_html=True)

    with c_sel:
        sel = st.selectbox("Language", list(LANG_MAP.keys()),
                           index=list(LANG_MAP.values()).index(lang),
                           key="landing_lang", label_visibility="collapsed")
        st.session_state.language = LANG_MAP[sel]

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Hero
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-pill">AI-Powered Fact Verification</div>
        <div class="t-hero">Truth is<br>non-negotiable.</div>
        <p class="hero-sub">
            {T('Verify claims, expose misinformation, and trace every story back to its source. '
               'Powered by global fact-checking databases and Vera, our AI reasoning analyst.', lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button(T("Start Fact-Checking", lang), use_container_width=True,
                     key="hero_cta", type="primary"):
            st.session_state.page = "app"; st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Features
    st.markdown("<p class='section-label'>Why TruthScope?</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='t-section'>{T('Built for a world full of noise', lang)}</div>",
                unsafe_allow_html=True)

    feats = [
        ("◎", T("Global Coverage",     lang), T("Verified claims from fact-checkers across 50+ countries", lang)),
        ("◈", T("AI Accuracy Match",   lang), T("Cosine-similarity scoring finds the closest verified claim", lang)),
        ("◉", T("8+ Languages",        lang), T("Verify in Hindi, Marathi, French, Spanish and more", lang)),
        ("◆", T("Meet Vera",           lang), T("Our AI analyst synthesises all evidence into a plain-language verdict", lang)),
        ("◇", T("Confidence Scores",   lang), T("Every result carries a transparent match percentage", lang)),
        ("◈", T("Trusted Sources",     lang), T("Only established, credentialed fact-checking organisations", lang)),
    ]
    st.markdown(
        "<div class='feat-grid'>" +
        "".join(f"<div class='feat-card'><span class='feat-icon'>{ic}</span>"
                f"<div class='feat-title'>{tt}</div><div class='feat-desc'>{dd}</div></div>"
                for ic, tt, dd in feats) +
        "</div>", unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # AI Engine card — two-panel split
    st.markdown("<p class='section-label'>AI Analyst</p>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ai-card">
        <div class="ai-card-left">
            <div class="ai-card-eyebrow">Meet your AI analyst</div>
            <div class="ai-card-name">Vera</div>
            <div class="ai-card-model">OpenAI GPT-OSS 120B via OpenRouter</div>
            <p class="ai-card-body">
                {T("Vera is TruthScope's built-in AI analyst. After the fact-check database returns "
                "matching records, Vera reads every result, weighs the credibility of each source, "
                "and writes a clear plain-language verdict. She operates on a two-turn reasoning "
                "chain: the first turn produces a full contextual analysis, the second distils it "
                "into a single definitive sentence.", lang)}
            </p>
            <div style="margin-top:14px;">
                <span class="ai-card-tag">120B parameters</span>
                <span class="ai-card-tag">Two-turn reasoning</span>
                <span class="ai-card-tag">OpenRouter API</span>
                <span class="ai-card-tag">GPT-OSS-120B</span>
            </div>
        </div>
        <div class="ai-card-right">
            <h5>{T('How Vera reasons', lang)}</h5>
            <p>{T('Vera receives the full claim text alongside every database match, its claimant, '
                  'the fact-checking publisher, the official rating, and the semantic match score.', lang)}</p>
            <p>{T('She then reasons over this evidence in context, identifying the most relevant '
                  'results and assessing source credibility before forming her analysis.', lang)}</p>
            <p>{T('In a second turn her own reasoning trace is passed back so she can confirm '
                  'and sharpen her conclusion into one direct verdict sentence.', lang)}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # How It Works
    st.markdown("<p class='section-label'>Process</p>", unsafe_allow_html=True)

    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.markdown(f"<div class='t-section' style='margin-bottom:8px;'>"
                    f"{T('How It Works', lang)}</div>", unsafe_allow_html=True)
    with col_btn:
        dive_label = T("Hide guide", lang) if st.session_state.show_deep_dive else T("In-depth guide", lang)
        if st.button(dive_label, key="deep_btn"):
            st.session_state.show_deep_dive = not st.session_state.show_deep_dive
            st.rerun()

    steps = [
        ("01", T("Paste a claim or headline", lang)),
        ("02", T("Google Fact Check API searches the database", lang)),
        ("03", T("AI ranks results by semantic similarity", lang)),
        ("04", T("Vera reasons over all evidence and delivers a verdict", lang)),
        ("05", T("Export the full report as a PDF", lang)),
    ]
    st.markdown(
        "<div class='steps-row'>" +
        "".join(f"<div class='step-item'><div class='step-num'>{n}</div>"
                f"<div class='step-text'>{tx}</div></div>" for n, tx in steps) +
        "</div>", unsafe_allow_html=True
    )

    if st.session_state.show_deep_dive:
        st.markdown(f"""
        <div class="deep-panel">
            <h4>What is TruthScope?</h4>
            <p>TruthScope is an AI-assisted fact-verification tool that cross-references user-submitted
            claims against the <strong>Google Fact Check Tools API</strong>, a curated index of
            fact-check articles published by professional organisations worldwide: PolitiFact, Snopes,
            AFP Fact Check, BBC Reality Check, Vishvas News, and hundreds more. All results are then
            processed by {AI_NAME}, TruthScope's AI analyst, who synthesises a final verdict.</p>

            <h4>Step 1: Input and Translation</h4>
            <p>You type a claim, headline, or any piece of text. For non-English languages, TruthScope
            uses <span class="mono-tag">deep-translator</span> backed by Google Translate to convert
            your input to English before querying the database, then translates all results back into
            your chosen language. Translations are cached with <span class="mono-tag">st.cache_data</span>
            so the same phrase is never translated twice in a session.</p>

            <h4>Step 2: Google Fact Check API Query</h4>
            <p>The translated claim is sent to the Google Fact Check Tools API which searches its
            full index of reviewed claims. The API returns matching claims together with the publisher
            that reviewed them, their verdict, and a URL to the full fact-check article.</p>

            <h4>Step 3: Semantic Similarity Scoring</h4>
            <p>Raw API results may include loosely related claims. TruthScope re-ranks them using
            <span class="mono-tag">sentence-transformers</span>
            (<span class="mono-tag">all-MiniLM-L6-v2</span>), which encodes both your query and each
            returned claim into semantic vectors and computes cosine similarity. Results are sorted
            highest to lowest so the most relevant match always appears first.</p>

            <h4>Step 4: {AI_NAME} AI Reasoning</h4>
            <p>All scored results are passed to {AI_NAME} (OpenAI GPT-OSS-120B via OpenRouter).
            The first turn produces a full contextual analysis of the evidence.
            {AI_NAME}'s reasoning trace is then preserved and passed into a second turn that asks
            her to confirm and distil her conclusion into a single definitive verdict sentence.
            This two-step approach produces more reliable and calibrated outputs than a single prompt.</p>

            <h4>Step 5: Rating Interpretation</h4>
            <p>Database ratings are colour-coded:
            <span class="mono-tag" style="background:#d5f5e3;color:#27ae60;">True</span>
            <span class="mono-tag" style="background:#fde8e8;color:#c0392b;">False</span>
            <span class="mono-tag" style="background:#fef3d0;color:#7d5632;">Misleading</span>
            <span class="mono-tag" style="background:#d4f1ee;color:#2a9d8f;">Unrated</span>
            </p>

            <h4>Step 6: Export</h4>
            <p>Each individual fact check, including {AI_NAME}'s analysis and verdict, can be saved
            as a styled PDF using Save as PDF. The Download session report button in the history panel
            exports every claim checked during your current session into a single branded document.</p>

            <h4>Limitations</h4>
            <p>TruthScope can only verify claims already reviewed by a professional fact-checking
            organisation and indexed by Google. Newly circulating claims, highly localised news, or
            niche topics may not appear. A no records found response does not mean a claim is true.
            {AI_NAME}'s analysis is a reasoning aid and not a substitute for primary source verification.</p>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button(T("Begin Verifying Now", lang), use_container_width=True,
                     key="cta2", type="primary"):
            st.session_state.page = "app"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# APP PAGE
# ══════════════════════════════════════════════════════════════════════════════
def app_page(lang):

    # Navbar: [Home btn] | [logo centered] | [Language label] | [selectbox]
    # Equal left/right weights so logo is truly centred
    c_back, c_logo, c_lbl, c_sel = st.columns([1.2, 5, 1, 1.4])

    with c_back:
        st.markdown("<div class='nav-cell nav-cell-left'>", unsafe_allow_html=True)
        if st.button("Home", key="back_btn"):
            st.session_state.page = "landing"
            st.session_state.results = None
            st.session_state.llm_analysis = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with c_logo:
        st.markdown("""
        <div class='nav-cell nav-cell-center'>
            <div class='nav-logo-wrap'>
                <div class='nav-logo'>Truth<span>Scope</span></div>
                <div class='nav-tagline'>AI-powered fact verification</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with c_lbl:
        st.markdown("<div class='nav-cell nav-cell-right'>"
                    "<span class='lang-lbl'>Language</span></div>",
                    unsafe_allow_html=True)

    with c_sel:
        sel = st.selectbox("Language", list(LANG_MAP.keys()),
                           index=list(LANG_MAP.values()).index(lang),
                           key="app_lang", label_visibility="collapsed")
        new_lang = LANG_MAP[sel]
        if new_lang != lang:
            st.session_state.language = new_lang; st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # API keys
    try:
        API_KEY = st.secrets["GOOGLE_FACT_CHECK_API_KEY"]
    except KeyError:
        st.error("Google Fact Check API key not found. Set GOOGLE_FACT_CHECK_API_KEY in secrets.toml.")
        st.stop()

    try:
        LLM_KEY = st.secrets["LLM_API"]
    except KeyError:
        LLM_KEY = None

    # Two-column layout
    col_main, col_hist = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown(f"<p class='section-label'>{T('Claim Verification', lang)}</p>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='t-section' style='font-size:1.3rem;margin-bottom:12px;'>"
            f"{T('What claim do you want to verify?', lang)}</div>",
            unsafe_allow_html=True
        )

        with st.form("fc_form", border=False):
            user_input = st.text_area(
                label="claim", height=105,
                placeholder=T("Paste a claim, headline, or news excerpt, e.g. Vaccines cause autism", lang),
                label_visibility="collapsed", key="textarea_input"
            )
            cs, cc = st.columns([3,1])
            with cs:
                submit = st.form_submit_button(
                    T("Verify Claim", lang), use_container_width=True, type="primary")
            with cc:
                clear = st.form_submit_button(T("Clear", lang), use_container_width=True)

        if clear:
            st.session_state.results = None
            st.session_state.llm_analysis = None
            st.session_state.last_query = ""
            st.rerun()

        if submit:
            q = user_input.strip()
            if not q:
                st.warning(T("Please enter some text before submitting.", lang))
            else:
                st.session_state.last_query = q
                en_q = T(q, "en") if lang != "en" else q

                with st.spinner(T("Searching fact-check databases...", lang)):
                    raw = fact_check(en_q, API_KEY)

                if raw == "no_results":
                    scored = []
                elif isinstance(raw, str) and raw.startswith("error:"):
                    st.error(T(f"Service unavailable: {raw[6:]}", lang))
                    scored = None
                else:
                    scored = add_scores(raw, en_q)

                st.session_state.results = scored if scored is not None else []

                analysis_text = None
                verdict_text  = None
                if LLM_KEY and scored is not None:
                    with st.spinner(T(f"{AI_NAME} is reasoning over the evidence...", lang)):
                        analysis_text, verdict_text = call_vera(q, scored or [], LLM_KEY)

                st.session_state.llm_analysis = (analysis_text, verdict_text)
                st.session_state.history.append({
                    "query":        q,
                    "results":      scored or [],
                    "llm_analysis": analysis_text,
                    "llm_verdict":  verdict_text,
                    "timestamp":    datetime.now().strftime("%H:%M"),
                })

    # History sidebar — all controls are full-width and vertically aligned
    with col_hist:
        st.markdown(
            f"<p class='section-label' style='margin-top:2px;'>"
            f"{T('Recent Checks', lang)}</p>",
            unsafe_allow_html=True
        )
        if st.session_state.history:
            # Clear history button — full width matches pdf button
            if st.button(T("Clear history", lang),
                         use_container_width=True, key="clear_hist"):
                st.session_state.history = []
                st.session_state.results = None
                st.session_state.llm_analysis = None
                st.rerun()

            # Session PDF — full-width styled button identical height to above
            sess_pdf = build_session_pdf(st.session_state.history)
            fname_s  = f"TruthScope_session_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            pdf_btn_sidebar(T("Download session report", lang), sess_pdf, fname_s)

            st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

            for item in reversed(st.session_state.history[-6:]):
                ts  = item.get("timestamp","")
                qry = item['query']
                n   = len(item.get('results') or [])
                tag = f"{n} result{'s' if n!=1 else ''}" if n else "no results"
                st.markdown(
                    f"<div class='hist-item'>"
                    f"<span style='font-family:\"DM Mono\",monospace;font-size:.6rem;color:#a07850;'>"
                    f"{ts} · {tag}</span><br>{qry[:34]}{'...' if len(qry)>34 else ''}"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f"<p class='t-caption'>{T('No recent checks yet', lang)}</p>",
                unsafe_allow_html=True
            )

    # Results section
    if st.session_state.results is not None:
        results      = st.session_state.results
        llm_pair     = st.session_state.llm_analysis
        analysis_txt = llm_pair[0] if llm_pair else None
        verdict_txt  = llm_pair[1] if llm_pair else None

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        if len(results) == 0 and not analysis_txt:
            st.markdown(f"""
            <div class="no-res-card">
                <div class="no-res-title">{T('No verified records found', lang)}</div>
                <div class="no-res-body">
                    {T('This claim has not yet been reviewed by a professional fact-checking '
                       'organisation in our database. That does not mean the claim is true. '
                       'It may be too recent, too localised, or simply not yet examined. '
                       'Try rephrasing with different keywords, or search a trusted source directly.', lang)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            ca, cb, cc = st.columns(3)
            with ca: st.link_button("Search Snopes",     "https://www.snopes.com/search/")
            with cb: st.link_button("Search PolitiFact", "https://www.politifact.com/search/")
            with cc: st.link_button("AFP Fact Check",    "https://factcheck.afp.com/")

        else:
            # Vera's analysis card
            if analysis_txt or verdict_txt:
                v_disp = T(verdict_txt,  lang) if verdict_txt  else ""
                a_disp = T(analysis_txt, lang) if analysis_txt else ""
                st.markdown(f"""
                <div class="vera-card">
                    <div class="vera-header">
                        <span class="vera-badge">{T(AI_NAME, lang)} · {T('AI Analysis', lang)}</span>
                        <span class="vera-model">OpenAI GPT-OSS-120B</span>
                    </div>
                    {"<div class='vera-verdict'>" + v_disp + "</div>" if v_disp else ""}
                    {"<div class='vera-body'>" + a_disp + "</div>" if a_disp else ""}
                </div>
                """, unsafe_allow_html=True)

            if results:
                st.markdown(
                    f"<p class='section-label' style='margin-top:16px;'>"
                    f"{T('Database Results', lang)} "
                    f"— {len(results)} {T('record(s) found', lang)}</p>",
                    unsafe_allow_html=True
                )

            q_now = st.session_state.last_query
            if q_now:
                pdf_bytes = build_single_pdf(q_now, results, analysis_txt, verdict_txt)
                fname_c   = f"TruthScope_{q_now[:28].replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
                pdf_btn_inline(T("Save as PDF", lang), pdf_bytes, fname_c)

            for idx, res in enumerate(results):
                pc  = rating_pill(res['rating'])
                bc  = sim_color(res['similarity_score'])
                cl  = T(res['claim'],        lang)
                ca_ = T(res['made_by'],      lang)
                ch  = T(res['fact_checker'], lang)
                rt  = T(res['rating'],       lang)

                st.markdown(f"""
                <div class="result-card" style="animation-delay:{idx*.07}s">
                    <span class="match-badge" style="background:{bc};">
                        {T('Match', lang)}: {res['similarity_score']}%
                    </span>
                    <div class="claim-text">{cl}</div>
                    <div class="meta-row">
                        <span class="meta-lbl">{T('Claimant', lang)}</span>
                        <span style="font-size:.84rem;">{ca_}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-lbl">{T('Verified by', lang)}</span>
                        <span style="font-size:.84rem;">{ch}</span>
                    </div>
                    <div class="meta-row" style="margin-top:7px;">
                        <span class="meta-lbl">{T('Rating', lang)}</span>
                        <span class="pill {pc}">{rt}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if res['source_link'] != '#':
                    st.link_button(T("Full Report", lang), res['source_link'])

                st.markdown("<div style='margin-bottom:3px;'></div>", unsafe_allow_html=True)


# ── ROUTER ────────────────────────────────────────────────────────────────────
if st.session_state.page == "landing":
    landing(st.session_state.language)
else:
    app_page(st.session_state.language)