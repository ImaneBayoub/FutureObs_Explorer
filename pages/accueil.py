"""
FUTURE-Obs — Page d'accueil
============================
Présente les trois points d'entrée de la plateforme et gère
la navigation initiale via st.session_state["page"].
"""

import streamlit as st
from config import ZONES


def page_accueil() -> None:
    st.markdown("""
    <div class="site-header">
        <div class="subtitle">Observatoire des littoraux · ANR FUTURE-Obs</div>
        <h1>FUTURE-Obs</h1>
        <div style="margin-top:0.8rem;font-family:'IBM Plex Sans',sans-serif;
                    font-size:0.9rem;color:#c8d8e8;line-height:1.7;max-width:700px">
            Plateforme d'analyse des interactions entre sociétés humaines et environnements
            littoraux à partir de contenus publics issus des réseaux sociaux.
            Corpus de <b style="color:#4fc3f7">842 000+ publications</b> collectées sur
            Facebook, Instagram, YouTube, TikTok, X et LinkedIn, autour de six parcs
            naturels marins français.
        </div>
    </div>
    """, unsafe_allow_html=True)

    btn_style = (
        "background:white;border:2px solid #4169E1;border-radius:12px;"
        "padding:1.5rem;text-align:center;cursor:pointer;"
        "transition:all 0.2s;box-shadow:0 2px 8px rgba(13,27,42,0.08);"
    )

    c1, c2, c3 = st.columns(3)

    # ── Données globales ──────────────────────────────────────────────────────
    with c1:
        st.markdown(
            f'<div style="{btn_style}">'
            '<div style="font-size:2rem"></div>'
            '<div style="font-family:\'Syne\',sans-serif;font-weight:700;'
            'font-size:1.1rem;margin:8px 0 4px">Données globales</div>'
            '<div style="font-size:0.82rem;color:#666">'
            'Corpus complet · filtres façade &amp; saison</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("→ Explorer le corpus global", key="btn_global",
                     use_container_width=True):
            st.session_state["page"] = "global"
            st.rerun()

    # ── Explorer un parc ──────────────────────────────────────────────────────
    with c2:
        st.markdown(
            f'<div style="{btn_style}">'
            '<div style="font-size:2rem"></div>'
            '<div style="font-family:\'Syne\',sans-serif;font-weight:700;'
            'font-size:1.1rem;margin:8px 0 4px">Explorer un parc</div>'
            '<div style="font-size:0.82rem;color:#666">'
            'Données par parc naturel marin</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        zone_choice = st.selectbox(
            "Zone d'étude", list(ZONES.keys()),
            key="home_zone", label_visibility="collapsed",
        )
        if st.button("→ Explorer ce parc", key="btn_parc",
                     use_container_width=True):
            st.session_state["page"] = "parc"
            st.session_state["zone"] = zone_choice
            st.rerun()

    # ── Comparer les parcs ────────────────────────────────────────────────────
    with c3:
        st.markdown(
            f'<div style="{btn_style}">'
            '<div style="font-size:2rem"></div>'
            '<div style="font-family:\'Syne\',sans-serif;font-weight:700;'
            'font-size:1.1rem;margin:8px 0 4px">Comparer les parcs</div>'
            '<div style="font-size:0.82rem;color:#666">'
            'Tops entités côte à côte · heatmaps</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("→ Comparer", key="btn_compare",
                     use_container_width=True):
            st.session_state["page"] = "compare"
            st.rerun()