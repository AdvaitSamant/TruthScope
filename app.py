import streamlit as st
import requests
import urllib.parse
import time
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FactSarkar - Fact Check with Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "FactSarkar - Verify facts with global fact-checking databases"
    }
)

# --- SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = []
if "results" not in st.session_state:
    st.session_state.results = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "language" not in st.session_state:
    st.session_state.language = "en"

# --- CACHE THE AI MODELS ---
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_translator():
    return Translator()

similarity_model = load_similarity_model()
translator = load_translator()

# --- LANGUAGE CONFIGURATION ---
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Chinese": "zh-cn",
    "Japanese": "ja"
}

def translate_text(text, lang):
    try:
        if lang == "en":
            return text
        return translator.translate(text, dest=lang).text
    except:
        return text

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Source+Serif+4:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --brown-50:  #fdf8f3;
        --brown-100: #f5ece0;
        --brown-200: #e8d5bc;
        --brown-300: #d4b896;
        --brown-400: #b8926a;
        --brown-500: #9b7145;
        --brown-600: #7d5632;
        --brown-700: #5c3d1e;
        --brown-800: #3d2710;
        --brown-900: #1f1208;
        --teal-400:  #2a9d8f;
        --teal-500:  #21867a;
        --teal-100:  #d4f1ee;
        --amber-400: #e9a825;
        --amber-100: #fef3d0;
        --red-400:   #c0392b;
        --red-100:   #fde8e8;
        --green-400: #27ae60;
        --green-100: #d5f5e3;
        --text-primary: #2c1a0e;
        --text-secondary: #6b4c33;
        --text-muted: #a07850;
        --surface: #fdf8f3;
        --card-bg: #fffcf8;
        --border: #e8d5bc;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--brown-50) !important;
        font-family: 'Source Serif 4', Georgia, serif;
        color: var(--text-primary);
    }

    /* Remove Streamlit chrome padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
    }

    [data-testid="stHeader"] {
        background-color: var(--brown-50) !important;
        border-bottom: 1px solid var(--border);
    }

    [data-testid="stSidebar"] {
        background-color: var(--brown-100) !important;
    }

    /* ---- TYPOGRAPHY SCALE ---- */
    .t-hero {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.8rem, 6vw, 4.5rem);
        font-weight: 800;
        color: var(--brown-800);
        letter-spacing: -1.5px;
        line-height: 1.05;
    }

    .t-section {
        font-family: 'Playfair Display', serif;
        font-size: clamp(1.6rem, 3vw, 2.2rem);
        font-weight: 700;
        color: var(--brown-700);
        letter-spacing: -0.5px;
        line-height: 1.2;
    }

    .t-card-title {
        font-family: 'Source Serif 4', serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.5;
    }

    .t-body {
        font-family: 'Source Serif 4', serif;
        font-size: 1rem;
        font-weight: 400;
        color: var(--text-secondary);
        line-height: 1.7;
    }

    .t-small {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted);
    }

    .t-caption {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-style: italic;
    }

    /* ---- HERO SECTION ---- */
    .hero-wrap {
        background: linear-gradient(135deg, var(--brown-100) 0%, var(--brown-50) 60%, #fff9f2 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 56px 48px 48px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
    }

    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, var(--brown-200) 0%, transparent 70%);
        opacity: 0.5;
        border-radius: 50%;
    }

    .hero-wrap::after {
        content: '"';
        position: absolute;
        bottom: -20px; right: 32px;
        font-family: 'Playfair Display', serif;
        font-size: 18rem;
        color: var(--brown-200);
        opacity: 0.4;
        line-height: 1;
    }

    .hero-tagline {
        display: inline-block;
        background: var(--amber-100);
        color: var(--brown-600);
        border: 1px solid var(--amber-400);
        border-radius: 100px;
        padding: 4px 14px;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 18px;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--text-secondary);
        max-width: 520px;
        line-height: 1.7;
        margin-top: 12px;
        font-weight: 300;
    }

    /* ---- FEATURE GRID ---- */
    .feat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin: 20px 0 8px;
    }

    .feat-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 22px 20px;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }

    .feat-card:hover {
        box-shadow: 0 6px 20px rgba(92, 61, 30, 0.1);
        transform: translateY(-2px);
    }

    .feat-icon {
        font-size: 1.7rem;
        margin-bottom: 10px;
        display: block;
    }

    .feat-title {
        font-family: 'Source Serif 4', serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }

    .feat-desc {
        font-size: 0.83rem;
        color: var(--text-muted);
        line-height: 1.55;
    }

    /* ---- HOW IT WORKS ---- */
    .steps-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0;
        background: var(--brown-100);
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
        margin: 12px 0 24px;
    }

    .step-item {
        padding: 22px 18px;
        text-align: center;
        border-right: 1px solid var(--border);
        position: relative;
    }

    .step-item:last-child { border-right: none; }

    .step-num {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--brown-300);
        line-height: 1;
        margin-bottom: 8px;
    }

    .step-text {
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }

    /* ---- DIVIDER ---- */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--brown-300), transparent);
        margin: 28px 0;
    }

    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--brown-400);
        margin-bottom: 10px;
    }

    /* ---- CTA BUTTON (Streamlit override) ---- */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--brown-600) 0%, var(--brown-500) 100%) !important;
        color: white !important;
        font-family: 'Source Serif 4', serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 2px 8px rgba(92, 61, 30, 0.25) !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, var(--brown-700) 0%, var(--brown-600) 100%) !important;
        box-shadow: 0 6px 18px rgba(92, 61, 30, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="baseButton-secondary"] {
        background: var(--brown-100) !important;
        color: var(--brown-700) !important;
        border: 1.5px solid var(--brown-300) !important;
        font-family: 'Source Serif 4', serif !important;
        font-size: 0.9rem !important;
        border-radius: 10px !important;
    }

    /* ---- INPUT STYLING ---- */
    textarea, input[type="text"] {
        background: var(--card-bg) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'Source Serif 4', serif !important;
        font-size: 0.95rem !important;
        color: var(--text-primary) !important;
    }

    textarea:focus, input[type="text"]:focus {
        border-color: var(--brown-400) !important;
        box-shadow: 0 0 0 3px rgba(155, 113, 69, 0.12) !important;
    }

    /* Hide the duplicate label on selectbox that causes overlap */
    [data-testid="stSelectbox"] label {
        display: none !important;
    }

    [data-testid="stSelectbox"] > div {
        margin-top: 0 !important;
    }

    /* ---- RESULT CARDS ---- */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .result-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-left: 4px solid var(--brown-400);
        border-radius: 12px;
        padding: 22px 24px 18px;
        margin-bottom: 14px;
        animation: fadeSlideUp 0.4s ease-out both;
        transition: box-shadow 0.25s ease;
    }

    .result-card:hover {
        box-shadow: 0 6px 20px rgba(92, 61, 30, 0.09);
    }

    .match-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 100px;
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: white;
        margin-bottom: 12px;
    }

    .claim-text {
        font-family: 'Source Serif 4', serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.55;
        margin-bottom: 12px;
    }

    .meta-row {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.83rem;
        color: var(--text-secondary);
        margin-bottom: 5px;
    }

    .meta-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: var(--text-muted);
        min-width: 80px;
    }

    /* ---- STATUS PILLS ---- */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 100px;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    .pill-false  { background: var(--red-100);   color: var(--red-400); }
    .pill-true   { background: var(--green-100);  color: var(--green-400); }
    .pill-mixed  { background: var(--amber-100);  color: var(--brown-600); }
    .pill-unrated{ background: var(--teal-100);   color: var(--teal-500); }

    /* ---- HISTORY SIDEBAR ---- */
    .history-item {
        background: var(--brown-100);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 7px;
        font-size: 0.8rem;
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ---- APP HEADER ---- */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 0 18px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 24px;
    }

    .app-logo {
        font-family: 'Playfair Display', serif;
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--brown-700);
        letter-spacing: -0.5px;
    }

    .app-logo span {
        color: var(--teal-400);
    }

    .app-tagline-sm {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-style: italic;
        margin-top: 2px;
    }

    /* ---- LANGUAGE SELECTOR WRAPPER ---- */
    .lang-wrap {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .lang-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--text-muted);
        white-space: nowrap;
    }

    /* Force selectbox to be compact */
    [data-testid="stSelectbox"] {
        margin-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---
def get_rating_pill(rating):
    r = rating.lower()
    if any(w in r for w in ['false', 'fake', 'pants on fire', 'incorrect']):
        return "pill-false"
    elif any(w in r for w in ['true', 'correct']):
        return "pill-true"
    elif any(w in r for w in ['mixture', 'misleading', 'half', 'partly']):
        return "pill-mixed"
    return "pill-unrated"

def get_similarity_color(score):
    if score >= 75: return "#27ae60"
    if score >= 50: return "#e9a825"
    return "#c0392b"

def check_fake_news(query, api_key):
    if not query.strip():
        return "Please enter text to verify."
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    url = f"{base_url}?query={urllib.parse.quote(query)}&key={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data or 'claims' not in data:
            return "No verified records found for this specific claim."
        results = []
        for claim in data['claims']:
            if 'claimReview' in claim and claim['claimReview']:
                review = claim['claimReview'][0]
                results.append({
                    "claim":       claim.get('text', 'No claim text provided'),
                    "made_by":     claim.get('claimant', 'Unknown Source'),
                    "fact_checker": review.get('publisher', {}).get('name', 'Unknown Publisher'),
                    "rating":      review.get('textualRating', 'Unrated'),
                    "source_link": review.get('url', '#')
                })
        return results
    except requests.exceptions.RequestException as e:
        return f"Service unavailable. (Error: {str(e)[:50]})"


# =====================================================================
# LANDING PAGE
# =====================================================================
def show_landing_page(language):
    # Top bar: logo + language
    col_logo, col_spacer, col_lang = st.columns([3, 3, 2])
    with col_logo:
        st.markdown("<div style='padding:8px 0;font-family:\"Playfair Display\",serif;font-size:1.6rem;font-weight:800;color:#5c3d1e;'>Fact<span style=\"color:#2a9d8f\">Sarkar</span></div>", unsafe_allow_html=True)
    with col_lang:
        st.markdown("<p class='lang-label' style='margin-bottom:2px;'>🌐 Language</p>", unsafe_allow_html=True)
        selected_lang = st.selectbox(
            "Language",
            list(lang_map.keys()),
            index=0 if language == "en" else list(lang_map.values()).index(language),
            key="landing_lang",
            label_visibility="collapsed"
        )
        st.session_state.language = lang_map[selected_lang]

    # Hero
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-tagline">Fact-Checking Intelligence</div>
        <div class="t-hero">Truth is<br>non-negotiable.</div>
        <p class="hero-subtitle">
            Verify claims, expose misinformation, and trace every story
            back to its source — powered by global fact-checking databases
            and AI-driven accuracy matching.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(translate_text("🔍  Start Fact-Checking", language), use_container_width=True, key="start_btn", type="primary"):
            st.session_state.page = "app"
            st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Features
    st.markdown("<p class='section-label'>Why FactSarkar?</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='t-section'>{translate_text('Built for a world full of noise', language)}</div>", unsafe_allow_html=True)

    features = [
        ("🌍", translate_text("Global Coverage", language),    translate_text("Verified claims from fact-checkers across 50+ countries", language)),
        ("🎯", translate_text("AI Accuracy Match", language),  translate_text("Cosine-similarity scoring finds the closest verified claim", language)),
        ("🌐", translate_text("8+ Languages", language),       translate_text("Verify claims in Hindi, Marathi, French, Spanish & more", language)),
        ("⚡", translate_text("Real-Time Results", language),  translate_text("Instant verification with direct links to source reports", language)),
        ("📊", translate_text("Confidence Scores", language),  translate_text("Every result comes with a transparent match percentage", language)),
        ("🔐", translate_text("Trusted Sources", language),    translate_text("Only established, credentialed fact-checking organisations", language)),
    ]

    st.markdown("<div class='feat-grid'>" + "".join(
        f"<div class='feat-card'><span class='feat-icon'>{ic}</span><div class='feat-title'>{t}</div><div class='feat-desc'>{d}</div></div>"
        for ic, t, d in features
    ) + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # How it works
    st.markdown("<p class='section-label'>Process</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='t-section' style='margin-bottom:14px;'>{translate_text('How It Works', language)}</div>", unsafe_allow_html=True)

    steps = [
        ("01", translate_text("Paste a claim or news excerpt", language)),
        ("02", translate_text("AI queries global fact-check databases", language)),
        ("03", translate_text("Results ranked by similarity score", language)),
        ("04", translate_text("Access full reports & source links", language)),
    ]
    st.markdown(
        "<div class='steps-row'>" +
        "".join(f"<div class='step-item'><div class='step-num'>{n}</div><div class='step-text'>{t}</div></div>" for n, t in steps) +
        "</div>",
        unsafe_allow_html=True
    )

    # Final CTA
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(translate_text("🚀  Begin Verifying Now", language), use_container_width=True, key="start_btn_2", type="primary"):
            st.session_state.page = "app"
            st.rerun()


# =====================================================================
# MAIN APP PAGE
# =====================================================================
def show_app_page(language):
    # Header row
    col_back, col_logo, col_lang = st.columns([1, 4, 1])

    with col_back:
        if st.button("← Home", key="back_btn"):
            st.session_state.page = "landing"
            st.session_state.results = None
            st.rerun()

    with col_logo:
        st.markdown("""
        <div style='text-align:center; padding: 6px 0;'>
            <div class='app-logo'>Fact<span>Sarkar</span></div>
            <div class='app-tagline-sm'>Global fact-checking at your fingertips</div>
        </div>
        """, unsafe_allow_html=True)

    with col_lang:
        st.markdown("<p class='lang-label' style='margin-bottom:2px;'>🌐 Language</p>", unsafe_allow_html=True)
        selected_lang = st.selectbox(
            "Language",
            list(lang_map.keys()),
            index=0 if language == "en" else list(lang_map.values()).index(language),
            key="app_lang",
            label_visibility="collapsed"
        )
        new_language = lang_map[selected_lang]
        if new_language != language:
            st.session_state.language = new_language
            st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # API Key
    try:
        API_KEY = st.secrets["GOOGLE_FACT_CHECK_API_KEY"]
    except KeyError:
        st.error("🔴 API key not found. Please set GOOGLE_FACT_CHECK_API_KEY in Streamlit secrets.")
        st.stop()

    # Two-column layout: main + history sidebar
    col_main, col_hist = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown(f"<p class='section-label'>{translate_text('Claim Verification', language)}</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='t-section' style='font-size:1.4rem; margin-bottom:16px;'>{translate_text('What claim do you want to verify?', language)}</div>", unsafe_allow_html=True)

        with st.form("fact_check_form", border=False):
            user_input = st.text_area(
                label="claim",
                height=110,
                placeholder=translate_text("Paste a claim, headline, or news excerpt — e.g. 'Vaccines cause autism'", language),
                value=st.session_state.user_input,
                label_visibility="collapsed"
            )

            col_submit, col_clear = st.columns([3, 1])
            with col_submit:
                submit_button = st.form_submit_button(
                    translate_text("✨  Verify Claim", language),
                    use_container_width=True,
                    type="primary"
                )
            with col_clear:
                clear_button = st.form_submit_button(translate_text("Clear", language), use_container_width=True)

        if clear_button:
            st.session_state.user_input = ""
            st.session_state.results = None
            st.rerun()

    with col_hist:
        st.markdown(f"<p class='section-label' style='margin-top:8px;'>{translate_text('Recent Checks', language)}</p>", unsafe_allow_html=True)
        if st.session_state.history:
            if st.button(translate_text("🗑️ Clear history", language), use_container_width=True, key="clear_history"):
                st.session_state.history = []
                st.rerun()
            for past_query in reversed(st.session_state.history[:6]):
                st.markdown(f"<div class='history-item'>→ {past_query[:38]}…</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='t-caption'>{translate_text('No recent checks yet', language)}</p>", unsafe_allow_html=True)

    # Processing
    if submit_button:
        st.session_state.user_input = user_input
        st.session_state.submitted = True

    if st.session_state.submitted:
        if not st.session_state.user_input.strip():
            st.warning(translate_text("⚠️ Please enter some text before submitting.", language))
        else:
            if st.session_state.user_input not in st.session_state.history:
                st.session_state.history.append(st.session_state.user_input)

            translated_query = translate_text(st.session_state.user_input, "en")

            with st.spinner(translate_text("🔍 Searching fact-check databases…", language)):
                results = check_fake_news(translated_query, API_KEY)

            if isinstance(results, str):
                st.session_state.results = None
                st.info(translate_text(results, language))
            else:
                query_embedding = similarity_model.encode(translated_query, convert_to_tensor=True)
                for res in results:
                    claim_embedding = similarity_model.encode(res['claim'], convert_to_tensor=True)
                    score = util.cos_sim(query_embedding, claim_embedding).item()
                    res['similarity_score'] = max(0, min(100, int(score * 100)))
                st.session_state.results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

        st.session_state.submitted = False

    # Results
    if st.session_state.results:
        results = st.session_state.results

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='section-label'>{translate_text('Results', language)} — "
            f"{len(results)} {translate_text('record(s) found', language)}</p>",
            unsafe_allow_html=True
        )

        for idx, res in enumerate(results):
            pill_class  = get_rating_pill(res['rating'])
            badge_color = get_similarity_color(res['similarity_score'])

            claim   = translate_text(res['claim'], language)
            claimant = translate_text(res['made_by'], language)
            checker = translate_text(res['fact_checker'], language)
            rating  = translate_text(res['rating'], language)

            st.markdown(f"""
            <div class="result-card" style="animation-delay:{idx * 0.07}s">
                <span class="match-badge" style="background:{badge_color};">
                    ● {translate_text('Match', language)}: {res['similarity_score']}%
                </span>
                <div class="claim-text">{claim}</div>
                <div class="meta-row">
                    <span class="meta-label">{translate_text('Claimant', language)}</span>
                    <span style="color:var(--text-secondary); font-size:0.87rem;">{claimant}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">{translate_text('Verified by', language)}</span>
                    <span style="color:var(--text-secondary); font-size:0.87rem;">{checker}</span>
                </div>
                <div class="meta-row" style="margin-top:10px;">
                    <span class="meta-label">{translate_text('Rating', language)}</span>
                    <span class="pill {pill_class}">{rating}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if res['source_link'] != '#':
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.link_button(translate_text("📄 Full Report", language), res['source_link'])

            st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)


# --- ROUTER ---
if st.session_state.page == "landing":
    show_landing_page(st.session_state.language)
else:
    show_app_page(st.session_state.language)