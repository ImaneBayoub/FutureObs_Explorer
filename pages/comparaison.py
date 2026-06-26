"""
FUTURE-Obs — Page comparaison des parcs
=========================================
Charge dynamiquement chaque parc via load_zone(),
affiche le UMAP multi-parcs, les tops entités côte à côte
et les heatmaps PMI Saison × Objets par parc.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import ZONE_COLORS, ZONES
from data.filters import compute_pmi, top_n
from data.loader import load_umap_html, load_zone
from ui.layout import header, section, show_umap


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[list[str], int]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔽 Parcs à comparer")
    selected = st.sidebar.multiselect(
        "Parcs", list(ZONES.keys()),
        default=list(ZONES.keys())[:3],
        key="cmp_zones",
    )
    top_n_cmp = st.sidebar.slider("Top N entités", 5, 30, 10, key="cmp_topn")
    return selected, top_n_cmp


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar_chart(df_top: pd.DataFrame, color: str, top_n_cmp: int) -> None:
    fig = px.bar(
        df_top, x="count", y="label", orientation="h",
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        height=max(250, top_n_cmp * 22 + 40),
        plot_bgcolor="#fafbfd", paper_bgcolor="#fafbfd",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _col_entity(zone_label: str, df: pd.DataFrame, label_col: str,
                color: str, top_n_cmp: int) -> None:
    st.markdown(f"**{zone_label}**")
    if df.empty or label_col not in df.columns:
        st.caption("(pas de données)")
        return
    df_top = top_n(df, label_col, top_n_cmp)
    _bar_chart(df_top, color, top_n_cmp)


def _heatmap_objets_saison(zone_label: str, obj_df: pd.DataFrame) -> None:
    """PMI objet × saison depuis ctx_objets (déjà précalculé)."""
    if obj_df.empty or "objet" not in obj_df.columns or "saison" not in obj_df.columns:
        return
    pmi_df = compute_pmi(obj_df, "objet", "saison")
    if pmi_df.empty:
        return
    mat = pmi_df.pivot(index="objet", columns="saison", values="pmi").fillna(0)
    with st.expander(f"**{zone_label}**", expanded=False):
        fig = px.imshow(
            mat, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0, zmin=-3, zmax=3,
            title=f"PMI Objet × Saison — {zone_label}",
        )
        fig.update_layout(
            height=max(300, len(mat) * 22 + 80),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page principale ───────────────────────────────────────────────────────────

def page_comparaison(token: str) -> None:
    selected, top_n_cmp = _render_sidebar()

    if len(selected) < 2:
        st.info("Sélectionnez au moins 2 parcs dans la sidebar.")
        return

    header(
        title="Comparaison des parcs",
        subtitle="FUTURE-Obs · Vue comparative",
        stats={"Parcs sélectionnés": len(selected)},
    )

    # ── UMAP multi-parcs ──────────────────────────────────────────────────────
    section("Espace sémantique — tous parcs")
    try:
        umap_html = load_umap_html("umap_dashboard_all_parcs.html", token)
        show_umap(umap_html, key="compare_umap",
                  caption="Projection UMAP — entités en gras = sur-représentées par parc")
    except Exception:
        st.info("Dashboard UMAP comparatif non disponible.")

    # ── Chargement des parcs ──────────────────────────────────────────────────
    section("Top entités par parc")
    with st.spinner("Chargement des parcs…"):
        parcs_data: dict[str, dict] = {
            zone_label: load_zone(ZONES[zone_label], token)
            for zone_label in selected
        }

    tabs_cmp = st.tabs([
        "🔵 Activités",
        "🔴 Impacts négatifs",
        "🟢 Impacts positifs",
        "🟣 Acteurs humains",
        "🔵 Acteurs non-humains",
    ])

    # ── Activités ─────────────────────────────────────────────────────────────
    with tabs_cmp[0]:
        cols = st.columns(len(selected))
        for i, zone_label in enumerate(selected):
            df_z  = parcs_data[zone_label].get("activites", pd.DataFrame())
            color = ZONE_COLORS.get(ZONES[zone_label], "#4169E1")
            with cols[i]:
                _col_entity(zone_label, df_z, "label_merged_act", color, top_n_cmp)

    # ── Impacts négatifs ──────────────────────────────────────────────────────
    with tabs_cmp[1]:
        cols = st.columns(len(selected))
        for i, zone_label in enumerate(selected):
            df_z = parcs_data[zone_label].get("impacts_neg", pd.DataFrame())
            with cols[i]:
                _col_entity(zone_label, df_z, "label_merged_imp", "#DC143C", top_n_cmp)

    # ── Impacts positifs ──────────────────────────────────────────────────────
    with tabs_cmp[2]:
        cols = st.columns(len(selected))
        for i, zone_label in enumerate(selected):
            df_z = parcs_data[zone_label].get("impacts_pos", pd.DataFrame())
            with cols[i]:
                _col_entity(zone_label, df_z, "label_merged_imp", "#3CB371", top_n_cmp)

    # ── Acteurs humains ───────────────────────────────────────────────────────
    with tabs_cmp[3]:
        cols = st.columns(len(selected))
        for i, zone_label in enumerate(selected):
            df_z = parcs_data[zone_label].get("acteurs_humain", pd.DataFrame())
            with cols[i]:
                _col_entity(zone_label, df_z, "label_merged_actor", "#9370DB", top_n_cmp)

    # ── Acteurs non-humains ───────────────────────────────────────────────────
    with tabs_cmp[4]:
        cols = st.columns(len(selected))
        for i, zone_label in enumerate(selected):
            df_z = parcs_data[zone_label].get("acteurs_non_humain", pd.DataFrame())
            with cols[i]:
                _col_entity(zone_label, df_z, "label_merged_actor", "#20B2AA", top_n_cmp)

    # ── Heatmaps PMI saison × objets ──────────────────────────────────────────
    section("Heatmap Saison × Objets par parc")
    for zone_label in selected:
        obj_z = parcs_data[zone_label].get("ctx_objets", pd.DataFrame())
        _heatmap_objets_saison(zone_label, obj_z)