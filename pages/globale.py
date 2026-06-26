"""
FUTURE-Obs — Page données globales
====================================
Charge les fichiers agrégés du global via load_global(),
applique les filtres objet / saison / facade via filter_agg(),
puis affiche les onglets avec les fonctions *_agg de tabs_shared.
"""

import pandas as pd
import streamlit as st

from config import FACADES_VALIDES, SAISONS
from data.filters import filter_agg
from data.loader import load_global, load_stats, load_umap_html
from ui.layout import header
from ui.tabs_shared import (
    tab_act_imp_agg,
    tab_acteurs_agg,
    tab_carte_agg,
    tab_objets_agg,
    tab_overview_umap,
    tab_pmi_agg,
)


# ── Sidebar filtres ───────────────────────────────────────────────────────────

def _render_sidebar_filters(ctx: dict) -> tuple[str | None, str | None, str | None]:
    """
    Filtres globaux — un seul choix à la fois (radio + selectbox).
    Saison et façade sont mutuellement exclusifs : choisir l'un désactive l'autre.
    La PMI n'est pas affectée par ces filtres (matrices précalculées).
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔽 Filtres")
    st.sidebar.caption("Un seul filtre actif à la fois — saison OU façade.")

    # Saison (radio — None = toutes)
    saison_opts = ["toutes"] + list(SAISONS)
    sel_saison_str = st.sidebar.radio(
        "Saison", saison_opts, index=0, key="g_saison",
    )
    sel_saison = None if sel_saison_str == "toutes" else sel_saison_str

    # Façade (seulement si pas de saison choisie)
    sel_facade = None
    if sel_saison is None:
        facade_opts = ["toutes"] + sorted(FACADES_VALIDES)
        sel_facade_str = st.sidebar.selectbox(
            "Façade maritime", facade_opts, index=0, key="g_facade",
        )
        sel_facade = None if sel_facade_str == "toutes" else sel_facade_str
    else:
        st.sidebar.selectbox(
            "Façade maritime", ["— (saison active)"], key="g_facade",
            disabled=True,
        )

    # Objet (un seul)
    obj_avail: list[str] = []
    obj_df = ctx.get("obj_saison", pd.DataFrame())
    if not obj_df.empty and "objet" in obj_df.columns:
        obj_avail = sorted(
            obj_df.loc[obj_df["objet"] != "__all__", "objet"]
            .dropna().unique().tolist()
        )
    sel_objet = None
    if obj_avail:
        obj_opts = ["tous"] + obj_avail
        sel_obj_str = st.sidebar.selectbox(
            "Objet détecté", obj_opts, index=0, key="g_objet",
        )
        sel_objet = None if sel_obj_str == "tous" else sel_obj_str

    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ Les filtres n'affectent pas l'onglet PMI (matrices précalculées).")

    return sel_saison, sel_facade, sel_objet


# ── Page principale ───────────────────────────────────────────────────────────

def page_globale(token: str) -> None:
    with st.spinner("Chargement des données globales…"):
        ctx = load_global(token)

    stats = load_stats("global")

    sel_saison, sel_facade, sel_objet = _render_sidebar_filters(ctx)

    # Filtrage : filter_agg gère le sentinel __all__ automatiquement
    ctx_f = filter_agg(ctx,
                       objets=[sel_objet] if sel_objet else None,
                       saisons=[sel_saison] if sel_saison else None,
                       facades=[sel_facade] if sel_facade else None)

    # Clé à utiliser selon le filtre actif (saison ou facade)
    ctx_key = "facade" if sel_facade else "saison"

    header(
        title="Données globales",
        subtitle="FUTURE-Obs · Corpus littoral complet",
        stats={
            "Posts":     stats.get("n_posts", 0),
            "Activités": int(ctx_f[f"act_{ctx_key}"]["n_posts"].sum())
                         if not ctx_f[f"act_{ctx_key}"].empty else 0,
            "Impacts":   int(ctx_f[f"imp_{ctx_key}"]["n_posts"].sum())
                         if not ctx_f[f"imp_{ctx_key}"].empty else 0,
            "Acteurs":   int(ctx_f[f"actor_{ctx_key}"]["n_posts"].sum())
                         if not ctx_f[f"actor_{ctx_key}"].empty else 0,
        },
    )

    tabs = st.tabs([
        "🔵 Vue d'ensemble",
        "🗺️ Carte",
        "📦 Objets",
        "⚡ Activités & Impacts",
        "👥 Acteurs",
        "📊 PMI",
    ])

    with tabs[0]:
        umap_html = ""
        try:
            umap_html = load_umap_html("umap_dashboard_all.html", token)
        except Exception:
            pass
        tab_overview_umap(umap_html, "Corpus global")

    with tabs[1]:
        tab_carte_agg(ctx_f, sel_saisons=[sel_saison] if sel_saison else None, sel_facades=[sel_facade] if sel_facade else None)

    with tabs[2]:
        tab_objets_agg(ctx_f)

    with tabs[3]:
        tab_act_imp_agg(ctx_f)

    with tabs[4]:
        tab_acteurs_agg(ctx_f)

    with tabs[5]:
        tab_pmi_agg(ctx_f)