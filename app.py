"""
FUTURE-Obs — Point d'entrée Streamlit
=======================================
Routing uniquement. Toute la logique est dans pages/, data/ et ui/.

Usage :
    streamlit run app.py
"""

import streamlit as st

from config import ZONES
from data.loader import resolve_token
from pages.accueil     import page_accueil
from pages.comparaison import page_comparaison
from pages.globale     import page_globale
from pages.parc        import page_parc
from ui.layout         import page_umap_fullscreen
from ui.sidebar        import render_nav

# ── Configuration de la page ──────────────────────────────────────────────────

st.set_page_config(
    page_title="FUTURE-Obs",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0d1b2a; border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #4fc3f7 !important; font-family: 'Syne', sans-serif !important; }
.main { background: #f7f9fc; }
.site-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 60%, #0a4f7a 100%);
    padding: 2rem 2.5rem 1.8rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.site-header h1 {
    font-family: 'Syne', sans-serif !important; font-weight: 800;
    font-size: 2.2rem; color: #ffffff !important; margin: 0 0 0.3rem 0;
}
.site-header .subtitle {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    color: #4fc3f7; letter-spacing: 2px; text-transform: uppercase;
}
.site-header .stats-row { display: flex; gap: 2.5rem; margin-top: 1.2rem; }
.stat-item { display: flex; flex-direction: column; }
.stat-number { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #4fc3f7; line-height: 1; }
.stat-label  { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #7aafd4; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 2px; }
.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: #0d1b2a; border-left: 4px solid #4fc3f7;
    padding-left: 12px; margin: 1.5rem 0 1rem;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #e8f0f7; padding: 4px; border-radius: 10px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    padding: 6px 18px; border-radius: 7px; color: #4a6080; background: transparent; border: none;
}
.stTabs [aria-selected="true"] { background: #0d1b2a !important; color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)


# ── Routing ───────────────────────────────────────────────────────────────────

def main() -> None:
    # Vue plein écran UMAP : prioritaire sur tout le reste
    if st.session_state.get("umap_fs_html"):
        page_umap_fullscreen()
        return

    token = resolve_token()

    # Sidebar : logo + bouton retour accueil (commun à toutes les pages)
    render_nav()

    page = st.session_state.get("page", "accueil")

    if page == "accueil":
        page_accueil()

    elif page == "global":
        page_globale(token)

    elif page == "parc":
        zone = st.session_state.get("zone", list(ZONES.keys())[0])
        page_parc(zone, token)

    elif page == "compare":
        page_comparaison(token)


if __name__ == "__main__":
    main()