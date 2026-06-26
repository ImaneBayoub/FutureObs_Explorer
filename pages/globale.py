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

def _render_sidebar_filters(ctx: dict) -> tuple[list, list, list]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔽 Filtres")

    sel_facades = st.sidebar.multiselect(
        "Façade maritime", FACADES_VALIDES, default=[], key="g_facade",
        placeholder="toutes les façades",
    )
    sel_saisons = st.sidebar.multiselect(
        "Saison", SAISONS, default=[], key="g_saison",
        placeholder="toutes les saisons",
    )

    # Liste des objets disponibles depuis ctx_objet_x_saison
    obj_avail: list[str] = []
    obj_df = ctx.get("obj_saison", pd.DataFrame())
    if not obj_df.empty and "objet" in obj_df.columns:
        obj_avail = sorted(
            obj_df.loc[obj_df["objet"] != "__all__", "objet"]
            .dropna().unique().tolist()
        )
    sel_objets = st.sidebar.multiselect(
        "Objet détecté", obj_avail, default=[], key="g_objet",
        placeholder="tous les objets",
    ) if obj_avail else []

    return sel_facades, sel_saisons, sel_objets


# ── Page principale ───────────────────────────────────────────────────────────

def page_globale(token: str) -> None:
    with st.spinner("Chargement des données globales…"):
        ctx = load_global(token)

    stats = load_stats("global")

    sel_facades, sel_saisons, sel_objets = _render_sidebar_filters(ctx)

    # Filtrage : filter_agg gère le sentinel __all__ automatiquement
    ctx_f = filter_agg(ctx,
                       objets=sel_objets or None,
                       saisons=sel_saisons or None,
                       facades=sel_facades or None)

    header(
        title="Données globales",
        subtitle="FUTURE-Obs · Corpus littoral complet",
        stats={
            "Posts":     stats.get("n_posts", 0),
            "Activités": int(ctx_f["act_saison"]["n_posts"].sum())
                         if not ctx_f["act_saison"].empty else 0,
            "Impacts":   int(ctx_f["imp_saison"]["n_posts"].sum())
                         if not ctx_f["imp_saison"].empty else 0,
            "Acteurs":   int(ctx_f["actor_hum_saison"]["n_posts"].sum()
                         + ctx_f["actor_nh_saison"]["n_posts"].sum())
                         if (not ctx_f["actor_hum_saison"].empty
                             or not ctx_f["actor_nh_saison"].empty) else 0,
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
        tab_carte_agg(ctx_f, sel_saisons=sel_saisons or None, sel_facades=sel_facades or None)

    with tabs[2]:
        tab_objets_agg(ctx_f)

    with tabs[3]:
        tab_act_imp_agg(ctx_f)

    with tabs[4]:
        tab_acteurs_agg(ctx_f)

    with tabs[5]:
        tab_pmi_agg(ctx_f)