"""
FUTURE-Obs — Page données globales
====================================
Charge la zone "global" depuis HuggingFace via load_zone(),
applique les filtres façade / saison / objet, puis affiche
les onglets Vue d'ensemble, Carte, Objets, Activités & Impacts,
Acteurs et PMI — même structure que les pages parc.
"""

import pandas as pd
import streamlit as st

from config import FACADES_VALIDES, SAISONS
from data.filters import filter_ctx
from data.loader import load_stats, load_umap_html, load_zone
from ui.layout import header
from ui.tabs_shared import (
    tab_act_imp,
    tab_acteurs,
    tab_carte,
    tab_objets,
    tab_overview_umap,
    tab_pmi,
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

    obj_avail: list[str] = []
    obj_df = ctx.get("ctx_objets", pd.DataFrame())
    if not obj_df.empty and "objet" in obj_df.columns:
        obj_avail = sorted(obj_df["objet"].dropna().unique().tolist())

    sel_objets = st.sidebar.multiselect(
        "Objet détecté", obj_avail, default=[], key="g_objet",
        placeholder="tous les objets",
    ) if obj_avail else []

    return sel_facades, sel_saisons, sel_objets


# ── Page principale ───────────────────────────────────────────────────────────

def page_globale(token: str) -> None:
    with st.spinner("Chargement des données globales…"):
        ctx = load_zone("global", token)

    stats = load_stats("global")

    sel_facades, sel_saisons, sel_objets = _render_sidebar_filters(ctx)

    # Renommage pour filter_ctx (attend la clé "objets")
    ctx_for_filter = {**ctx, "objets": ctx.get("ctx_objets", pd.DataFrame())}
    ctx_f = filter_ctx(ctx_for_filter, sel_facades, sel_saisons, sel_objets)

    acts_df       = ctx_f.get("activites",        pd.DataFrame())
    imp_neg_df    = ctx_f.get("impacts_neg",       pd.DataFrame())
    imp_pos_df    = ctx_f.get("impacts_pos",       pd.DataFrame())
    imp_neu_df    = ctx_f.get("impacts_neutre",    pd.DataFrame())
    actrs_hum_df  = ctx_f.get("acteurs_humain",    pd.DataFrame())
    actrs_nh_df   = ctx_f.get("acteurs_non_humain",pd.DataFrame())
    locs_df       = ctx_f.get("localisations",     pd.DataFrame())
    obj_df        = ctx_f.get("objets",            pd.DataFrame())

    # DataFrame acteurs combiné pour les onglets qui l'utilisent en entier
    actrs_df = pd.concat(
        [df for df in [actrs_hum_df, actrs_nh_df] if not df.empty],
        ignore_index=True,
    ) if (not actrs_hum_df.empty or not actrs_nh_df.empty) else pd.DataFrame()

    # DataFrame impacts combiné pour tab_act_imp (camembert tonalité)
    imps_combined = pd.concat(
        [df for df in [imp_neg_df, imp_pos_df, imp_neu_df] if not df.empty],
        ignore_index=True,
    ) if any(not df.empty for df in [imp_neg_df, imp_pos_df, imp_neu_df]) else pd.DataFrame()

    header(
        title="Données globales",
        subtitle="FUTURE-Obs · Corpus littoral complet",
        stats={
            "Posts":      stats.get("n_posts", 0),
            "Activités":  len(acts_df),
            "Impacts −":  len(imp_neg_df),
            "Impacts +":  len(imp_pos_df),
            "Acteurs":    len(actrs_df),
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
        tab_carte(locs_df, pd.DataFrame(), acts_df, imps_combined, actrs_df)

    with tabs[2]:
        tab_objets(ctx_objets=obj_df)

    with tabs[3]:
        tab_act_imp(acts_df, imp_neg_df, imp_pos_df, imp_neu_df)

    with tabs[4]:
        tab_acteurs(actrs_hum_df, actrs_nh_df)

    with tabs[5]:
        tab_pmi(
            ctx_acts    = acts_df,
            ctx_imp_neg = imp_neg_df,
            ctx_imp_pos = imp_pos_df,
            ctx_actrs   = actrs_df,
            label_act   = "label_merged_act",
            label_imp   = "label_merged_imp",
            label_actr  = "label_merged_actor",
            ctx_objets  = obj_df,
        )