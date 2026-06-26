"""
FUTURE-Obs — Page exploration d'un parc naturel marin
======================================================
Charge la zone parc depuis HuggingFace via load_zone(),
puis expose les onglets Vue d'ensemble, Carte, Objets,
Activités & Impacts, Acteurs, PMI, Corpus et Export.
"""

import pandas as pd
import streamlit as st

from config import ZONES
from data.loader import load_stats, load_umap_html, load_zone
from ui.layout import header
from ui.tabs_shared import (
    tab_act_imp,
    tab_acteurs,
    tab_carte,
    tab_export,
    tab_objets,
    tab_overview_umap,
    tab_pmi,
)

_UMAP_PARC_PATHS: dict[str, str] = {
    "mediterranee_sites": "umap_dashboard_mediterranee_sites.html",
}


# ── Page principale ───────────────────────────────────────────────────────────

def page_parc(zone_label: str, token: str) -> None:
    parc_slug = ZONES[zone_label]

    with st.spinner("Chargement des données du parc…"):
        ctx = load_zone(parc_slug, token)

    stats = load_stats(parc_slug)

    acts_df      = ctx.get("activites",         pd.DataFrame())
    imp_neg_df   = ctx.get("impacts_neg",        pd.DataFrame())
    imp_pos_df   = ctx.get("impacts_pos",        pd.DataFrame())
    imp_neu_df   = ctx.get("impacts_neutre",     pd.DataFrame())
    actrs_hum_df = ctx.get("acteurs_humain",     pd.DataFrame())
    actrs_nh_df  = ctx.get("acteurs_non_humain", pd.DataFrame())
    locs_df      = ctx.get("localisations",      pd.DataFrame())
    obj_df       = ctx.get("ctx_objets",         pd.DataFrame())

    # DataFrames combinés pour les onglets qui les utilisent en entier
    actrs_df = pd.concat(
        [df for df in [actrs_hum_df, actrs_nh_df] if not df.empty],
        ignore_index=True,
    ) if (not actrs_hum_df.empty or not actrs_nh_df.empty) else pd.DataFrame()

    imps_combined = pd.concat(
        [df for df in [imp_neg_df, imp_pos_df, imp_neu_df] if not df.empty],
        ignore_index=True,
    ) if any(not df.empty for df in [imp_neg_df, imp_pos_df, imp_neu_df]) else pd.DataFrame()

    header(
        title=zone_label,
        subtitle="FUTURE-Obs · Parc naturel marin",
        stats={
            "Posts":         stats.get("n_posts", 0),
            "Activités":     len(acts_df),
            "Impacts":       len(imps_combined),
            "Acteurs":       len(actrs_df),
            "Localisations": len(locs_df),
        },
    )

    tabs = st.tabs([
        "🔵 Vue d'ensemble",
        "🗺️ Carte",
        "📦 Objets",
        "⚡ Activités & Impacts",
        "👥 Acteurs",
        "📊 PMI",
        "⬇️ Export",
    ])

    with tabs[0]:
        umap_html = ""
        try:
            umap_path = _UMAP_PARC_PATHS.get(
                parc_slug,
                f"umap_dashboard_simple_{parc_slug}.html",
            )
            umap_html = load_umap_html(umap_path, token)
        except Exception:
            pass
        tab_overview_umap(umap_html, zone_label)

    with tabs[1]:
        tab_carte(locs_df, pd.DataFrame(), acts_df, imps_combined, actrs_df)

    with tabs[2]:
        tab_objets(ctx_objets=obj_df)

    with tabs[3]:
        tab_act_imp(acts_df, imps_combined)

    with tabs[4]:
        tab_acteurs(actrs_df)

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

    with tabs[6]:
        tab_export(ctx, zone_label)