"""
FUTURE-Obs — Logique sidebar
==============================
  render_nav()             : logo + bouton retour accueil
  render_sidebar_global()  : filtres façade / saison / objet
  render_sidebar_parc()    : sélecteur de parc
  render_sidebar_compare() : sélecteur multi-parcs + slider top-N
"""

import pandas as pd
import streamlit as st

from config import FACADES_VALIDES, SAISONS, ZONES


# ── Navigation commune ────────────────────────────────────────────────────────

def render_nav() -> None:
    with st.sidebar:
        st.markdown(
            "<h1 style=\"font-family:'Syne',sans-serif;"
            "font-size:1.4rem;color:#4fc3f7\">🌊 FUTURE-Obs</h1>",
            unsafe_allow_html=True,
        )
        page = st.session_state.get("page", "accueil")
        if page != "accueil":
            if st.button("← Accueil", key="btn_home_sidebar"):
                st.session_state["page"] = "accueil"
                st.rerun()


# ── Filtres page globale ──────────────────────────────────────────────────────

def render_sidebar_global(ctx_objets: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Paramètres
    ----------
    ctx_objets : DataFrame avec colonne "objet" (depuis load_zone("global")).
                 Utilisé pour peupler la liste des objets disponibles.

    Retourne
    --------
    (sel_facades, sel_saisons, sel_objets) — listes vides = pas de filtre actif.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔽 Filtres")

        sel_facades: list[str] = st.multiselect(
            "Façade maritime", FACADES_VALIDES, default=[], key="g_facade",
            placeholder="toutes les façades",
        )
        sel_saisons: list[str] = st.multiselect(
            "Saison", SAISONS, default=[], key="g_saison",
            placeholder="toutes les saisons",
        )

        # Objets disponibles depuis ctx_objets (clé "ctx_objets" dans load_zone)
        obj_avail: list[str] = []
        if not ctx_objets.empty and "objet" in ctx_objets.columns:
            obj_avail = sorted(ctx_objets["objet"].dropna().unique().tolist())

        sel_objets: list[str] = st.multiselect(
            "Objet détecté", obj_avail, default=[], key="g_objet",
            placeholder="tous les objets",
        ) if obj_avail else []

    return sel_facades, sel_saisons, sel_objets


# ── Sélecteur de parc ─────────────────────────────────────────────────────────

def render_sidebar_parc() -> str:
    with st.sidebar:
        st.markdown("---")
        zone_now = st.session_state.get("zone", list(ZONES.keys())[0])
        new_zone: str = st.selectbox(
            "Changer de parc",
            list(ZONES.keys()),
            index=list(ZONES.keys()).index(zone_now),
            key="sidebar_zone",
        )
        if new_zone != zone_now:
            st.session_state["zone"] = new_zone
            st.rerun()
    return st.session_state.get("zone", list(ZONES.keys())[0])


# ── Sélecteur comparaison ─────────────────────────────────────────────────────

def render_sidebar_compare() -> tuple[list[str], int]:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔽 Parcs à comparer")
        selected: list[str] = st.multiselect(
            "Parcs", list(ZONES.keys()),
            default=list(ZONES.keys())[:3],
            key="cmp_zones",
        )
        top_n_cmp: int = st.slider("Top N entités", 5, 30, 10, key="cmp_topn")
    return selected, top_n_cmp