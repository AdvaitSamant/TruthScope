import streamlit as st
import requests
import urllib.parse
import time
from sentence_transformers import SentenceTransformer, util

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FactSarkar",
    layout="centered",
    initial_sidebar_state="collapsed"
)
if 'history' not in st.session_state:
    st.session_state.history = []


# --- CACHE THE AI MODEL ---
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

similarity_model = load_similarity_model()

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
        animation: fadeIn 0.4s ease-out;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .result-card {
            background-color: #1e1e1e;
            border: 1px solid #333333;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
    }
    
    .result-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        border-color: #cccccc;
    }
    
    .claim-text {
        font-size: 1.15em;
        font-weight: 600;
        color: #111111;
        margin-bottom: 12px;
        line-height: 1.4;
    }
    
    @media (prefers-color-scheme: dark) {
        .claim-text { color: #eeeeee; }
    }
    
    .meta-data {
        font-size: 0.9em;
        color: #555555;
        margin-bottom: 6px;
    }
    
    @media (prefers-color-scheme: dark) {
        .meta-data { color: #aaaaaa; }
    }
    
    .similarity-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: 600;
        color: white;
        margin-bottom: 16px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# --- CORE LOGIC ---
def check_fake_news(query, api_key):
    if not query.strip():
        return "Please enter text to verify."

    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    safe_query = urllib.parse.quote(query)
    url = f"{base_url}?query={safe_query}&key={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'claims' not in data:
            return "No verified records found for this specific claim."
        
        results = []
        for claim in data['claims']:
            claim_text = claim.get('text', 'No claim text provided')
            claimant = claim.get('claimant', 'Unknown Source')
            
            if 'claimReview' in claim and len(claim['claimReview']) > 0:
                review = claim['claimReview'][0]
                publisher = review.get('publisher', {}).get('name', 'Unknown Publisher')
                rating = review.get('textualRating', 'Unrated')
                article_url = review.get('url', '#')
                
                results.append({
                    "claim": claim_text,
                    "made_by": claimant,
                    "fact_checker": publisher,
                    "rating": rating,
                    "source_link": article_url
                })
        return results

    except requests.exceptions.RequestException as e:
        return f"Service currently unavailable. (Error: {e})"

# --- HELPER UI FUNCTIONS ---
def get_rating_color(rating):
    rating_lower = rating.lower()
    if any(word in rating_lower for word in ['false', 'fake', 'pants on fire', 'incorrect']):
        return "red"
    elif any(word in rating_lower for word in ['true', 'correct']):
        return "green"
    elif any(word in rating_lower for word in ['mixture', 'misleading', 'half true', 'partly']):
        return "orange"
    return "blue"

def get_similarity_color(score):
    if score >= 0.75: return "#2e7d32" # Deep green
    if score >= 0.50: return "#ed6c02" # Deep orange
    return "#d32f2f" # Deep red

# --- APP LAYOUT ---
st.title("FactSarkar")
st.markdown("Verify news, articles, and public claims against global fact-checking databases. FactSarkar utilizes semantic AI to match the context of your query with verified reports.")

# Safely get API key
try:
    API_KEY = st.secrets["API_KEY"]
except KeyError:
    st.error("API key not found. Please set your API key in Streamlit secrets.")
    st.stop()

# User Input
with st.form("fact_check_form"):
    user_input = st.text_area(
        "Enter the claim or news excerpt to verify:", 
        height=120, 
        placeholder="Enter the statement here..."
    )
    submit_button = st.form_submit_button("Verify Claim", use_container_width=True)

with st.sidebar:
    st.header("Search History")
    
    # 1. Show the Clear History button right below the heading
    if st.session_state.history:
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun() # Instantly refreshes the app
            
    st.divider()

    # 2. Display the history list just once
    if not st.session_state.history:
        st.write("No previous searches.")
    else:
        for past_query in reversed(st.session_state.history):
            st.caption(f" {past_query}")

# Processing and Results
if submit_button:
    if not user_input.strip():
        st.warning("Please enter some text before submitting.")
    else:
        # --- NEW: Save to history if it's not a duplicate ---
        if user_input not in st.session_state.history:
            st.session_state.history.append(user_input)

        with st.spinner("Querying databases and analyzing semantics..."):
            time.sleep(0.5) 
            results = check_fake_news(user_input, API_KEY)

        if isinstance(results, str):
            st.info(results)
        else:
            # Calculate Semantic Similarity
            query_embedding = similarity_model.encode(user_input, convert_to_tensor=True)
            
            for res in results:
                claim_embedding = similarity_model.encode(res['claim'], convert_to_tensor=True)
                cosine_score = util.cos_sim(query_embedding, claim_embedding).item()
                res['similarity_score'] = max(0, min(100, int(cosine_score * 100)))

            # Sort results by similarity descending
            results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

            st.success(f"Found {len(results)} verified record(s).")
            st.divider()
            
            # Display Results
            for res in results:
                rating_color = get_rating_color(res['rating'])
                sim_color = get_similarity_color(res['similarity_score'] / 100)
                
                card_html = f"""
                <div class="result-card">
                    <span class="similarity-badge" style="background-color: {sim_color};">
                        Match Confidence: {res['similarity_score']}%
                    </span>
                    <div class="claim-text">Claim: {res['claim']}</div>
                    <div class="meta-data"><b>Claimant:</b> {res['made_by']}</div>
                    <div class="meta-data"><b>Verified By:</b> {res['fact_checker']}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Final Assessment:** :{rating_color}[{res['rating']}]")
                with col2:
                    if res['source_link'] != '#':
                        st.link_button("Read Full Report", res['source_link'], use_container_width=True)
                
                st.write("")
