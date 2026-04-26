import streamlit as st
import requests
import urllib.parse
import base64
import io
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
    page_title="FactScope – Fact-Check with Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "FactScope – Verify facts with global fact-checking databases"}
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in {
    'history': [],       # list of {query, results, timestamp}
    'results': None,     # current results (list or [])
    'last_query': "",
    'page': "landing",
    'language': "en",
    'show_deep_dive': False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ── LANGUAGE ──────────────────────────────────────────────────────────────────
LANG_MAP = {
    "English": "en", "Hindi": "hi", "Marathi": "mr",
    "French": "fr",  "German": "de", "Spanish": "es",
    "Chinese (Simplified)": "zh-CN", "Japanese": "ja"
}

@st.cache_data(ttl=3600, show_spinner=False)
def tr(text: str, lang: str) -> str:
    """Translate text to lang; cached so repeated calls are instant."""
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

/* Give the page breathing room at the top */
.block-container{
  padding-top:2.8rem!important;
  padding-bottom:2.5rem!important;
  max-width:1100px!important;
}

[data-testid="stHeader"]{background:var(--bg)!important;border-bottom:1px solid var(--border);}
[data-testid="stSidebar"]{background:var(--b100)!important;}

/* ── NAVBAR ── */
.nav-logo{
  font-family:'Playfair Display',serif;
  font-size:1.6rem;font-weight:800;
  color:var(--b700);letter-spacing:-.5px;line-height:1;
  padding-top:8px;
}
.nav-logo span{color:var(--teal);}
.nav-tagline{font-size:.76rem;color:var(--tm);font-style:italic;margin-top:1px;}

.lang-lbl{
  font-family:'DM Mono',monospace;font-size:.62rem;font-weight:500;
  letter-spacing:1.6px;text-transform:uppercase;color:var(--tm);
  padding-top:14px;white-space:nowrap;
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
.hero-sub{font-size:1.05rem;color:var(--ts);max-width:490px;line-height:1.75;margin-top:10px;font-weight:300;}

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

/* ── STEPS ROW ── */
.steps-row{
  display:grid;grid-template-columns:repeat(4,1fr);
  background:var(--b100);border:1px solid var(--border);
  border-radius:11px;overflow:hidden;margin:8px 0 16px;
}
.step-item{padding:18px 14px;text-align:center;border-right:1px solid var(--border);}
.step-item:last-child{border-right:none;}
.step-num{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:800;color:var(--b300);line-height:1;margin-bottom:5px;}
.step-text{font-size:.78rem;color:var(--ts);line-height:1.4;}

/* ── DEEP-DIVE PANEL ── */
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

/* ── BUTTONS ── */
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
/* Tertiary (Ghost) Buttons */
[data-testid="baseButton-tertiary"]{
  background:transparent!important;
  color:var(--b600)!important;
  font-family:'Source Serif 4',serif!important;
  font-size:.9rem!important;font-weight:600!important;
  border:none!important;
  padding:0!important;
  transition:color .2s!important;
}
[data-testid="baseButton-tertiary"]:hover{
  color:var(--b800)!important;
  background:transparent!important;
  text-decoration:underline;
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

/* hide duplicate selectbox label */
[data-testid="stSelectbox"] label{display:none!important;}
[data-testid="stSelectbox"]>div{margin-top:0!important;}

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

.no-res-card{
  background:var(--b100);border:1px dashed var(--b300);
  border-radius:11px;padding:26px 26px;
  text-align:center;margin:6px 0;
}
.no-res-title{
  font-family:'Playfair Display',serif;
  font-size:1.05rem;font-weight:700;color:var(--b700);margin-bottom:7px;
}
.no-res-body{font-size:.85rem;color:var(--ts);line-height:1.65;max-width:460px;margin:0 auto;}

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

/* ── HISTORY ── */
.hist-item{
  background:var(--b100);border:1px solid var(--border);
  border-radius:7px;padding:7px 10px;margin-bottom:5px;
  font-size:.78rem;color:var(--ts);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}

/* ── PDF BUTTON ── */
.pdf-btn a button{
  background:linear-gradient(135deg,#7d5632,#9b7145);
  color:white;border:none;
  padding:7px 16px;border-radius:8px;
  font-family:'Source Serif 4',serif;
  font-size:.85rem;font-weight:600;
  cursor:pointer;
  box-shadow:0 2px 8px rgba(92,61,30,.22);
  transition:all .2s;
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
    q = model.encode(query_text, convert_to_tensor=True)
    for res in results:
        c = model.encode(res['claim'], convert_to_tensor=True)
        res['similarity_score'] = max(0, min(100, int(util.cos_sim(q, c).item() * 100)))
    return sorted(results, key=lambda x: x['similarity_score'], reverse=True)


# ── PDF BUILDERS ──────────────────────────────────────────────────────────────

def _doc_styles():
    br   = colors.HexColor("#5c3d1e")
    muted= colors.HexColor("#a07850")
    sec  = colors.HexColor("#6b4c33")
    light= colors.HexColor("#f5ece0")
    S    = getSampleStyleSheet()
    return br, muted, sec, light, S

def build_single_pdf(query, results):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
          leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    br, muted, sec, light, S = _doc_styles()

    heading = ParagraphStyle('H', fontName='Times-Bold',   fontSize=22, textColor=br,   spaceAfter=3,  leading=26)
    sub     = ParagraphStyle('S', fontName='Times-Italic', fontSize=10, textColor=muted, spaceAfter=14, leading=14)
    lbl     = ParagraphStyle('L', fontName='Courier-Bold', fontSize=7.5,textColor=muted, spaceAfter=1,  leading=10)
    claim   = ParagraphStyle('C', fontName='Times-Bold',   fontSize=11, textColor=colors.HexColor("#2c1a0e"), spaceAfter=8, leading=16)
    val     = ParagraphStyle('V', fontName='Times-Roman',  fontSize=9.5,textColor=sec,   spaceAfter=5,  leading=14)
    foot    = ParagraphStyle('F', fontName='Times-Italic', fontSize=8,  textColor=muted, alignment=TA_CENTER)

    story = [
        Paragraph("FactScope", heading),
        Paragraph(f"Fact Check Report &nbsp;·&nbsp; {datetime.now().strftime('%d %B %Y, %H:%M')}", sub),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e8d5bc"), spaceAfter=10),
        Paragraph("QUERY", lbl),
        Paragraph(query, claim),
        HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#e8d5bc"), spaceAfter=12),
    ]

    for i, res in enumerate(results, 1):
        story.append(Paragraph(f"RESULT {i}", lbl))
        story.append(Paragraph(res['claim'], claim))
        data = [
            ["Match",      f"{res['similarity_score']}%"],
            ["Claimant",   res['made_by']],
            ["Verified by",res['fact_checker']],
            ["Rating",     res['rating']],
            ["Source",     res['source_link']],
        ]
        tbl = Table(data, colWidths=[3.2*cm, 13.3*cm])
        tbl.setStyle(TableStyle([
            ('FONTNAME',  (0,0),(0,-1),'Courier-Bold'),
            ('FONTSIZE',  (0,0),(0,-1), 7),
            ('TEXTCOLOR', (0,0),(0,-1), muted),
            ('FONTNAME',  (1,0),(1,-1),'Times-Roman'),
            ('FONTSIZE',  (1,0),(1,-1), 9.5),
            ('TEXTCOLOR', (1,0),(1,-1), sec),
            ('VALIGN',    (0,0),(-1,-1),'TOP'),
            ('ROWBACKGROUNDS',(0,0),(-1,-1),[light, colors.white]),
            ('TOPPADDING',(0,0),(-1,-1), 4),
            ('BOTTOMPADDING',(0,0),(-1,-1), 4),
            ('LEFTPADDING',(0,0),(-1,-1), 6),
        ]))
        story.append(tbl)
        story.append(Spacer(1, .45*cm))
        if i < len(results):
            story.append(HRFlowable(width="100%", thickness=.5,
                          color=colors.HexColor("#e8d5bc"), spaceAfter=10))

    story += [Spacer(1, .7*cm),
              HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#e8d5bc"), spaceAfter=5),
              Paragraph("Generated by FactScope", foot)]
    doc.build(story)
    buf.seek(0)
    return buf.read()


def build_session_pdf(history):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
          leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    br, muted, sec, light, S = _doc_styles()

    heading = ParagraphStyle('H', fontName='Times-Bold',   fontSize=22, textColor=br,   spaceAfter=3,  leading=26)
    sub     = ParagraphStyle('S', fontName='Times-Italic', fontSize=10, textColor=muted, spaceAfter=14, leading=14)
    lbl     = ParagraphStyle('L', fontName='Courier-Bold', fontSize=7,  textColor=muted, spaceAfter=1,  leading=10)
    q_style = ParagraphStyle('Q', fontName='Times-Bold',   fontSize=11, textColor=colors.HexColor("#2c1a0e"), spaceAfter=6, leading=16)
    val     = ParagraphStyle('V', fontName='Times-Roman',  fontSize=9,  textColor=sec,   spaceAfter=4,  leading=13)
    foot    = ParagraphStyle('F', fontName='Times-Italic', fontSize=8,  textColor=muted, alignment=TA_CENTER)

    story = [
        Paragraph("FactScope", heading),
        Paragraph(f"Session Report &nbsp;·&nbsp; {datetime.now().strftime('%d %B %Y, %H:%M')} &nbsp;·&nbsp; {len(history)} check(s)", sub),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e8d5bc"), spaceAfter=14),
    ]

    for ci, check in enumerate(history, 1):
        story.append(Paragraph(f"CHECK {ci}", lbl))
        story.append(Paragraph(check['query'], q_style))
        if not check['results']:
            story.append(Paragraph("No verified records found.", val))
        else:
            for i, res in enumerate(check['results'], 1):
                data = [
                    ["#",        str(i)],
                    ["Match",    f"{res['similarity_score']}%"],
                    ["Claim",    res['claim']],
                    ["Claimant", res['made_by']],
                    ["Checker",  res['fact_checker']],
                    ["Rating",   res['rating']],
                    ["Source",   res['source_link']],
                ]
                tbl = Table(data, colWidths=[2.5*cm, 14*cm])
                tbl.setStyle(TableStyle([
                    ('FONTNAME',  (0,0),(0,-1),'Courier-Bold'),
                    ('FONTSIZE',  (0,0),(0,-1), 7),
                    ('TEXTCOLOR', (0,0),(0,-1), muted),
                    ('FONTNAME',  (1,0),(1,-1),'Times-Roman'),
                    ('FONTSIZE',  (1,0),(1,-1), 9),
                    ('TEXTCOLOR', (1,0),(1,-1), sec),
                    ('VALIGN',    (0,0),(-1,-1),'TOP'),
                    ('ROWBACKGROUNDS',(0,0),(-1,-1),[light, colors.white]),
                    ('TOPPADDING',(0,0),(-1,-1), 3),
                    ('BOTTOMPADDING',(0,0),(-1,-1), 3),
                    ('LEFTPADDING',(0,0),(-1,-1), 5),
                ]))
                story.append(tbl)
                story.append(Spacer(1, .2*cm))
        story.append(Spacer(1, .35*cm))
        if ci < len(history):
            story.append(HRFlowable(width="100%", thickness=.5,
                          color=colors.HexColor("#e8d5bc"), spaceAfter=8))

    story += [Spacer(1, .7*cm),
              HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#e8d5bc"), spaceAfter=5),
              Paragraph("Generated by FactScope", foot)]
    doc.build(story)
    buf.seek(0)
    return buf.read()


def pdf_dl_button(label, pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(
        f'<div class="pdf-btn"><a href="data:application/pdf;base64,{b64}" '
        f'download="{filename}"><button>{label}</button></a></div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def landing(lang):

    # ── Top bar ──────────────────────────────────────────
    c_logo, c_space, c_lbl, c_sel = st.columns([3, 3, 0.85, 1.35])
    with c_logo:
        st.markdown(
            "<div class='nav-logo' style='padding-top:8px;'>Fact<span>Scope</span></div>",
            unsafe_allow_html=True
        )
    with c_lbl:
        st.markdown("<p class='lang-lbl'>Language</p>", unsafe_allow_html=True)
    with c_sel:
        sel = st.selectbox("Language", list(LANG_MAP.keys()),
                           index=list(LANG_MAP.values()).index(lang),
                           key="landing_lang", label_visibility="collapsed")
        st.session_state.language = LANG_MAP[sel]

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-pill">Fact-Checking Intelligence</div>
        <div class="t-hero">Truth is<br>non-negotiable.</div>
        <p class="hero-sub">
            {T('Verify claims, expose misinformation, and trace every story back to its source — powered by global fact-checking databases and AI accuracy matching.', lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button(T("Start Fact-Checking", lang), use_container_width=True,
                     key="hero_cta", type="primary"):
            st.session_state.page = "app"; st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Features ─────────────────────────────────────────
    st.markdown("<p class='section-label'>Why FactScope?</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='t-section'>{T('Built for a world full of noise', lang)}</div>",
                unsafe_allow_html=True)

    feats = [
        ("◎", T("Global Coverage",   lang), T("Verified claims from fact-checkers across 50+ countries", lang)),
        ("◈", T("AI Accuracy Match", lang), T("Cosine-similarity scoring finds the closest verified claim", lang)),
        ("◉", T("8+ Languages",      lang), T("Verify in Hindi, Marathi, French, Spanish & more", lang)),
        ("◆", T("Real-Time Results", lang), T("Instant verification with links to primary source reports", lang)),
        ("◇", T("Confidence Scores", lang), T("Every result carries a transparent match percentage", lang)),
        ("◈", T("Trusted Sources",   lang), T("Only established, credentialed fact-checking organisations", lang)),
    ]
    st.markdown(
        "<div class='feat-grid'>" +
        "".join(f"<div class='feat-card'><span class='feat-icon'>{ic}</span>"
                f"<div class='feat-title'>{tt}</div><div class='feat-desc'>{dd}</div></div>"
                for ic, tt, dd in feats) +
        "</div>", unsafe_allow_html=True
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── How It Works ─────────────────────────────────────
    st.markdown("<p class='section-label'>Process</p>", unsafe_allow_html=True)

    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.markdown(f"<div class='t-section' style='margin-bottom:8px;'>{T('How It Works', lang)}</div>",
                    unsafe_allow_html=True)
    with col_btn:
        st.markdown("<div style='padding-top:6px;'></div>", unsafe_allow_html=True)
        dive_label = T("Hide details", lang) if st.session_state.show_deep_dive else T("In-depth guide", lang)
        
        # Add type="tertiary" here
        if st.button(dive_label, key="deep_btn", type="tertiary"):
            st.session_state.show_deep_dive = not st.session_state.show_deep_dive
            st.rerun()

    steps = [
        ("01", T("Paste a claim or news excerpt", lang)),
        ("02", T("AI queries global fact-check databases", lang)),
        ("03", T("Results ranked by similarity score", lang)),
        ("04", T("Access full reports and source links", lang)),
    ]
    st.markdown(
        "<div class='steps-row'>" +
        "".join(f"<div class='step-item'><div class='step-num'>{n}</div>"
                f"<div class='step-text'>{tx}</div></div>" for n, tx in steps) +
        "</div>", unsafe_allow_html=True
    )

    if st.session_state.show_deep_dive:
        st.markdown(
            "<div class='deep-panel'>"
            "<h4>What is FactScope?</h4>"
            "<p>FactScope is an AI-assisted fact-verification tool that cross-references user-submitted claims against the <strong>Google Fact Check Tools API</strong> — a curated index of fact-check articles published by professional organisations worldwide: PolitiFact, Snopes, AFP Fact Check, BBC Reality Check, Vishvas News, and hundreds more.</p>"
            
            "<h4>Step 1 — Input &amp; Translation</h4>"
            "<p>You type a claim, headline, or any piece of text. For non-English languages, FactScope uses <span class='mono-tag'>deep-translator</span> (backed by Google Translate) to convert your input to English before querying the database — then translates all results back into your chosen language. All translations are cached with <span class='mono-tag'>@st.cache_data</span> so the same phrase is never translated twice in a session, keeping things fast.</p>"
            
            "<h4>Step 2 — API Query</h4>"
            "<p>The translated claim is sent to the Google Fact Check Tools API, which searches its full index of reviewed claims. The API returns matching claims along with the publisher that reviewed them, their verdict, and a URL to the full fact-check article.</p>"
            
            "<h4>Step 3 — AI Similarity Scoring</h4>"
            "<p>Raw API results may include loosely related claims. FactScope re-ranks them using <span class='mono-tag'>sentence-transformers</span> (<span class='mono-tag'>all-MiniLM-L6-v2</span>), which encodes both your query and each returned claim into semantic vectors and computes cosine similarity. Results are sorted highest-to-lowest so the most relevant match always appears first.</p>"
            
            "<h4>Step 4 — Rating Interpretation</h4>"
            "<p>Ratings are colour-coded by verdict type: "
            "<span class='mono-tag' style='background:#d5f5e3;color:#27ae60;'>True / Correct</span> "
            "<span class='mono-tag' style='background:#fde8e8;color:#c0392b;'>False / Fake</span> "
            "<span class='mono-tag' style='background:#fef3d0;color:#7d5632;'>Misleading / Mixture</span> "
            "<span class='mono-tag' style='background:#d4f1ee;color:#2a9d8f;'>Unrated / Unknown</span></p>"
            
            "<h4>Step 5 — Export</h4>"
            "<p>Each individual fact check can be saved as a styled PDF using <em>Save as PDF</em>. The <em>Download session report</em> button in the history panel exports every claim checked during your current session into a single document.</p>"
            
            "<h4>Limitations</h4>"
            "<p>FactScope can only verify claims already reviewed by a professional fact-checking organisation and indexed by Google. Newly circulating claims, highly localised news, or niche topics may not appear. A &quot;No records found&quot; response does not mean a claim is true — it means it has not yet been formally reviewed. In that case, try rephrasing or check Snopes, PolitiFact, or AFP directly.</p>"
            "</div>",
            unsafe_allow_html=True
        )

    # ── Final CTA ────────────────────────────────────────
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button(T("Begin Verifying Now", lang), use_container_width=True,
                     key="cta2", type="primary"):
            st.session_state.page = "app"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# APP PAGE
# ══════════════════════════════════════════════════════════════════════════════
def app_page(lang):

    # ── Navbar ───────────────────────────────────────────
    c_back, c_logo, c_sp, c_lbl, c_sel = st.columns([1.1, 3, 2.5, 0.85, 1.35])

    with c_back:
        st.markdown("<div style='padding-top:6px;'></div>", unsafe_allow_html=True)
        if st.button("← Home", key="back_btn"):
            st.session_state.page = "landing"
            st.session_state.results = None
            st.rerun()

    with c_logo:
        st.markdown("""
        <div style='text-align:center;padding:4px 0 0;'>
            <div class='nav-logo'>Fact<span>Scope</span></div>
            <div class='nav-tagline'>Global fact-checking at your fingertips</div>
        </div>""", unsafe_allow_html=True)

    with c_lbl:
        st.markdown("<p class='lang-lbl'>Language</p>", unsafe_allow_html=True)

    with c_sel:
        sel = st.selectbox("Language", list(LANG_MAP.keys()),
                           index=list(LANG_MAP.values()).index(lang),
                           key="app_lang", label_visibility="collapsed")
        new_lang = LANG_MAP[sel]
        if new_lang != lang:
            st.session_state.language = new_lang; st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── API Key ──────────────────────────────────────────
    try:
        API_KEY = st.secrets["GOOGLE_FACT_CHECK_API_KEY"]
    except KeyError:
        st.error("API key not found. Set GOOGLE_FACT_CHECK_API_KEY in Streamlit secrets.")
        st.stop()

    # ── Layout ───────────────────────────────────────────
    col_main, col_hist = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown(f"<p class='section-label'>{T('Claim Verification', lang)}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='t-section' style='font-size:1.3rem;margin-bottom:12px;'>"
            f"{T('What claim do you want to verify?', lang)}</div>",
            unsafe_allow_html=True
        )

        with st.form("fc_form", border=False):
            user_input = st.text_area(
                label="claim",
                height=105,
                placeholder=T("Paste a claim, headline, or news excerpt — e.g. 'Vaccines cause autism'", lang),
                label_visibility="collapsed",
                key="textarea_input"
            )
            cs, cc = st.columns([3,1])
            with cs:
                submit = st.form_submit_button(T("Verify Claim", lang), use_container_width=True, type="primary")
            with cc:
                clear  = st.form_submit_button(T("Clear", lang), use_container_width=True)

        if clear:
            st.session_state.results = None
            st.session_state.last_query = ""
            st.rerun()

        if submit:
            q = user_input.strip()
            if not q:
                st.warning(T("Please enter some text before submitting.", lang))
            else:
                st.session_state.last_query = q
                # Translate to English only if needed — cached so instant on repeats
                en_q = T(q, "en") if lang != "en" else q

                with st.spinner(T("Searching fact-check databases...", lang)):
                    raw = fact_check(en_q, API_KEY)

                if raw == "no_results":
                    st.session_state.results = []
                    st.session_state.history.append({
                        "query": q, "results": [], "timestamp": datetime.now().strftime("%H:%M")
                    })
                elif isinstance(raw, str) and raw.startswith("error:"):
                    st.session_state.results = None
                    st.error(T(f"Service unavailable: {raw[6:]}", lang))
                else:
                    scored = add_scores(raw, en_q)
                    st.session_state.results = scored
                    st.session_state.history.append({
                        "query": q, "results": scored, "timestamp": datetime.now().strftime("%H:%M")
                    })

    with col_hist:
        st.markdown(
            f"<p class='section-label' style='margin-top:2px;'>{T('Recent Checks', lang)}</p>",
            unsafe_allow_html=True
        )
        if st.session_state.history:
            if st.button(T("Clear history", lang), use_container_width=True, key="clear_hist"):
                st.session_state.history = []
                st.session_state.results = None
                st.rerun()

            # Session PDF download
            sess_pdf = build_session_pdf(st.session_state.history)
            fname_s  = f"FactScope_session_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            pdf_dl_button("Download session report", sess_pdf, fname_s)
            st.markdown("<div style='margin-bottom:9px;'></div>", unsafe_allow_html=True)

            for item in reversed(st.session_state.history[-6:]):
                ts  = item.get("timestamp","")
                qry = item['query']
                n   = len(item.get('results') or [])
                tag = f"{n} result{'s' if n!=1 else ''}" if n else "no results"
                st.markdown(
                    f"<div class='hist-item'>"
                    f"<span style='font-family:\"DM Mono\",monospace;font-size:.6rem;color:#a07850;'>"
                    f"{ts} · {tag}</span><br>{qry[:34]}{'…' if len(qry)>34 else ''}"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"<p class='t-caption'>{T('No recent checks yet', lang)}</p>", unsafe_allow_html=True)

    # ── Results ──────────────────────────────────────────
    if st.session_state.results is not None:
        results = st.session_state.results
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        if len(results) == 0:
            st.markdown(f"""
            <div class="no-res-card">
                <div class="no-res-title">{T('No verified records found', lang)}</div>
                <div class="no-res-body">
                    {T('This claim has not yet been reviewed by a professional fact-checking organisation in our database. '
                       'That does not mean the claim is true — it may be too recent, too localised, or not yet examined. '
                       'Try rephrasing with different keywords, or search a trusted source directly.', lang)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            ca, cb, cc = st.columns(3)
            with ca: st.link_button("Search Snopes",     "https://www.snopes.com/search/")
            with cb: st.link_button("Search PolitiFact", "https://www.politifact.com/search/")
            with cc: st.link_button("AFP Fact Check",    "https://factcheck.afp.com/")

        else:
            st.markdown(
                f"<p class='section-label'>{T('Results', lang)} — "
                f"{len(results)} {T('record(s) found', lang)}</p>",
                unsafe_allow_html=True
            )

            # Save current check as PDF
            q_now = st.session_state.last_query
            if q_now:
                pdf_bytes = build_single_pdf(q_now, results)
                fname_c   = f"FactScope_{q_now[:28].replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
                pdf_dl_button("Save as PDF", pdf_bytes, fname_c)
                st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

            for idx, res in enumerate(results):
                pc  = rating_pill(res['rating'])
                bc  = sim_color(res['similarity_score'])
                cl  = T(res['claim'],        lang)
                ca_ = T(res['made_by'],       lang)
                ch  = T(res['fact_checker'],  lang)
                rt  = T(res['rating'],        lang)

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