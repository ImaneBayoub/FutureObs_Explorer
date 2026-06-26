"""
FUTURE-Obs — Onglets partagés
==============================
Toutes les fonctions d'onglet réutilisées entre page_globale et page_parc.
Aucune fonction ici ne connaît la page qui l'appelle.

  tab_overview_umap  : espace sémantique (iframe UMAP)
  tab_objets         : distribution des objets + PMI saison
  tab_act_imp        : activités et impacts (top-N + camembert tonalité)
  tab_acteurs        : top acteurs colorés par type + camembert humain/non-humain
  tab_pmi            : matrices PMI entité × saison ou façade
  tab_carte          : carte Scattermap des localisations
  tab_export         : téléchargement ZIP des CSV
"""

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import TOP_N
from data.filters import compute_pmi, top_n
from ui.layout import section, show_umap


# ── Constantes visuelles locales ──────────────────────────────────────────────

_BG     = "#fafbfd"
_MARGIN = dict(l=10, r=10, t=10, b=10)
_TYPE_COLORS = {"humain": "#9370DB", "non humain": "#20B2AA"}


# ── Vue d'ensemble (UMAP) ─────────────────────────────────────────────────────

def tab_overview_umap(umap_html: str, label: str) -> None:
    section("Espace sémantique")
    if umap_html:
        show_umap(
            umap_html,
            key=label.replace(" ", "_"),
            caption=f"Projection UMAP — {label}",
        )
    else:
        st.info("Dashboard UMAP non disponible.")


# ── Objets ────────────────────────────────────────────────────────────────────

def tab_objets(ctx_objets: pd.DataFrame | None = None) -> None:
    """
    Affiche la distribution des objets et une matrice PMI objet × saison.
    Source : ctx_objets précalculé (colonne "objet") — toujours disponible
    depuis les fichiers ctx_objets.csv générés par precompute_all.py.
    """
    section("Distribution des objets détectés")

    if ctx_objets is None or ctx_objets.empty or "objet" not in ctx_objets.columns:
        st.info("Aucune donnée d'objets disponible.")
        return

    obj_counts = (
        ctx_objets["objet"].dropna()
        .value_counts()
        .reset_index()
    )
    obj_counts.columns = ["Objet", "Occurrences"]

    df_for_pmi = (
        ctx_objets[["objet", "saison"]].dropna()
        if "saison" in ctx_objets.columns
        else pd.DataFrame()
    )

    c1, c2 = st.columns([3, 2])

    with c1:
        fig = px.bar(
            obj_counts, x="Occurrences", y="Objet", orientation="h",
            color="Occurrences",
            color_continuous_scale=["#c8ddf5", "#0d1b2a"],
        )
        fig.update_layout(
            height=max(300, len(obj_counts) * 22 + 60),
            plot_bgcolor=_BG, paper_bgcolor=_BG,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            margin=_MARGIN,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if not df_for_pmi.empty:
            st.markdown("**Objets × Saison — PMI**")
            pmi_df = compute_pmi(df_for_pmi, "objet", "saison")
            if not pmi_df.empty:
                mat = pmi_df.pivot(
                    index="objet", columns="saison", values="pmi"
                ).fillna(0)
                fig2 = px.imshow(
                    mat, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0, zmin=-3, zmax=3,
                )
                fig2.update_layout(
                    height=max(350, len(mat) * 22 + 80),
                    margin=_MARGIN,
                )
                st.plotly_chart(fig2, use_container_width=True)


# ── Activités & Impacts ───────────────────────────────────────────────────────

def tab_act_imp(
    acts:     pd.DataFrame,
    imps_neg: pd.DataFrame,
    imps_pos: pd.DataFrame,
    imps_neu: pd.DataFrame | None = None,
    label_act: str = "label_merged_act",
    label_imp: str = "label_merged_imp",
) -> None:
    """
    Reçoit directement les DataFrames déjà séparés par polarité
    (impacts_neg, impacts_pos, impacts_neutre) — plus de filtrage à la volée.
    """
    section("Activités & Impacts")

    if imps_neu is None:
        imps_neu = pd.DataFrame()

    # Impacts combinés pour le camembert de tonalité
    imps_all = pd.concat(
        [df for df in [imps_neg, imps_pos, imps_neu] if not df.empty],
        ignore_index=True,
    )

    # Sliders
    n_act = max(5, acts[label_act].dropna().nunique()) if not acts.empty and label_act in acts.columns else 5
    n_imp = max(5, max(
        (len(df) for df in [imps_neg, imps_pos, imps_neu] if not df.empty),
        default=5,
    ))

    col1, col2 = st.columns(2)
    with col1:
        top_n_act = st.slider("Top N activités", 1, n_act, min(20, n_act),
                               key=f"sl_act_{id(acts)}")
    with col2:
        top_n_imp = st.slider("Top N impacts", 1, n_imp, min(10, n_imp),
                               key=f"sl_imp_{id(imps_neg)}")

    c1, c2 = st.columns(2)

    # ── Colonne gauche : activités + camembert ────────────────────────────────
    with c1:
        st.markdown("**🔵 Activités**")
        if not acts.empty and label_act in acts.columns:
            df_top = top_n(acts, label_act, top_n_act)
            fig = px.bar(
                df_top, x="count", y="label", orientation="h",
                color="count", color_continuous_scale=["#c8d8f5", "#4169E1"],
            )
            fig.update_layout(
                height=max(300, len(df_top) * 22 + 40),
                plot_bgcolor=_BG, paper_bgcolor=_BG,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                margin=_MARGIN,
            )
            st.plotly_chart(fig, use_container_width=True)

        if not imps_all.empty:
            st.markdown("**Répartition des impacts**")
            # Reconstituer la tonalité depuis les DataFrames séparés
            tone_data = []
            for df, nom in [(imps_neg, "négatif"), (imps_pos, "positif"), (imps_neu, "neutre")]:
                if not df.empty:
                    tone_data.append({"Tonalité": nom, "Nb": len(df)})
            tone = pd.DataFrame(tone_data)
            fig2 = px.pie(
                tone, names="Tonalité", values="Nb", hole=0.45,
                color="Tonalité",
                color_discrete_map={
                    "positif": "#3CB371",
                    "négatif": "#DC143C",
                    "neutre":  "#9e9e9e",
                },
            )
            fig2.update_layout(height=220, margin=_MARGIN)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Colonne droite : impacts neg + pos ────────────────────────────────────
    with c2:
        st.markdown("**🔴 Impacts négatifs**")
        if not imps_neg.empty and label_imp in imps_neg.columns:
            df_neg = top_n(imps_neg, label_imp, top_n_imp)
            fig = px.bar(
                df_neg, x="count", y="label", orientation="h",
                color="count", color_continuous_scale=["#fcd0d0", "#DC143C"],
            )
            fig.update_layout(
                height=max(250, len(df_neg) * 22 + 40),
                plot_bgcolor=_BG, paper_bgcolor=_BG,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                margin=_MARGIN,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**🟢 Impacts positifs**")
        if not imps_pos.empty and label_imp in imps_pos.columns:
            df_pos = top_n(imps_pos, label_imp, top_n_imp)
            fig = px.bar(
                df_pos, x="count", y="label", orientation="h",
                color="count", color_continuous_scale=["#d4f5e2", "#3CB371"],
            )
            fig.update_layout(
                height=max(250, len(df_pos) * 22 + 40),
                plot_bgcolor=_BG, paper_bgcolor=_BG,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                margin=_MARGIN,
            )
            st.plotly_chart(fig, use_container_width=True)

        if not imps_neu.empty and label_imp in imps_neu.columns:
            st.markdown("**⚪ Impacts neutres**")
            df_neu = top_n(imps_neu, label_imp, top_n_imp)
            fig = px.bar(
                df_neu, x="count", y="label", orientation="h",
                color="count", color_continuous_scale=["#e8e8e8", "#9e9e9e"],
            )
            fig.update_layout(
                height=max(200, len(df_neu) * 22 + 40),
                plot_bgcolor=_BG, paper_bgcolor=_BG,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                margin=_MARGIN,
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Acteurs ───────────────────────────────────────────────────────────────────

def tab_acteurs(
    actrs_humain:     pd.DataFrame,
    actrs_non_humain: pd.DataFrame,
    label_col: str = "label_merged_actor",
) -> None:
    """
    Reçoit directement les DataFrames séparés humain / non-humain.
    """
    section("Acteurs identifiés")

    if actrs_humain.empty and actrs_non_humain.empty:
        st.info("Aucune donnée d'acteurs.")
        return

    actrs_all = pd.concat(
        [df for df in [actrs_humain, actrs_non_humain] if not df.empty],
        ignore_index=True,
    )

    c1, c2 = st.columns([3, 2])

    with c1:
        df_top = top_n(actrs_all, label_col, TOP_N)
        # Couleur par type : humain = violet, non-humain = teal
        hum_labels = set(
            actrs_humain[label_col].str.lower().dropna().unique()
            if not actrs_humain.empty and label_col in actrs_humain.columns
            else []
        )
        bar_colors = [
            _TYPE_COLORS["humain"] if lbl in hum_labels else _TYPE_COLORS["non humain"]
            for lbl in df_top["label"]
        ]
        fig = go.Figure(go.Bar(
            x=df_top["count"], y=df_top["label"],
            orientation="h",
            marker_color=bar_colors,
        ))
        fig.update_layout(
            height=520,
            plot_bgcolor=_BG, paper_bgcolor=_BG,
            showlegend=False,
            yaxis=dict(autorange="reversed"),
            margin=_MARGIN,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tc = pd.DataFrame([
            {"Type": "humain",     "Nb": len(actrs_humain)},
            {"Type": "non humain", "Nb": len(actrs_non_humain)},
        ])
        fig2 = px.pie(
            tc, names="Type", values="Nb",
            color="Type", color_discrete_map=_TYPE_COLORS,
            hole=0.5, title="Humain vs non-humain",
        )
        fig2.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ── PMI ───────────────────────────────────────────────────────────────────────

def tab_pmi(
    ctx_acts:    pd.DataFrame,
    ctx_imp_neg: pd.DataFrame,
    ctx_imp_pos: pd.DataFrame,
    ctx_actrs:   pd.DataFrame,
    label_act:   str = "label_merged_act",
    label_imp:   str = "label_merged_imp",
    label_actr:  str = "label_merged_actor",
    ctx_objets:  pd.DataFrame | None = None,
) -> None:
    section("Associations — PMI")

    mode = st.radio(
        "Contexte", ["Saison", "Façade maritime"],
        horizontal=True, key=f"pmi_mode_{id(ctx_acts)}",
    )
    ctx_col = "saison" if mode == "Saison" else "meta_facade"

    sub_tabs = st.tabs([
        "🔵 Activités",
        "🔴 Impacts négatifs",
        "🟢 Impacts positifs",
        "🟣 Acteurs",
        "📦 Objets",
    ])

    def _show_pmi(df: pd.DataFrame, lcol: str, tab_key: str) -> None:
        pmi_df = compute_pmi(df, lcol, ctx_col)
        if pmi_df.empty:
            st.info("Pas assez de données.")
            return
        mat = pmi_df.pivot(index=lcol, columns=ctx_col, values="pmi").fillna(0)
        top_ents = df[lcol].dropna().str.lower().value_counts().head(30).index
        mat = mat[mat.index.isin(top_ents)]
        fig = px.imshow(
            mat, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0, zmin=-3, zmax=3,
        )
        fig.update_layout(
            height=max(400, len(mat) * 18 + 80),
            margin=_MARGIN,
        )
        st.plotly_chart(fig, use_container_width=True, key=tab_key)

    with sub_tabs[0]:
        _show_pmi(ctx_acts,    label_act,  f"pmi_act_{id(ctx_acts)}")
    with sub_tabs[1]:
        _show_pmi(ctx_imp_neg, label_imp,  f"pmi_neg_{id(ctx_imp_neg)}")
    with sub_tabs[2]:
        _show_pmi(ctx_imp_pos, label_imp,  f"pmi_pos_{id(ctx_imp_pos)}")
    with sub_tabs[3]:
        _show_pmi(ctx_actrs,   label_actr, f"pmi_actr_{id(ctx_actrs)}")
    with sub_tabs[4]:
        if ctx_objets is not None and not ctx_objets.empty and "objet" in ctx_objets.columns:
            _show_pmi(ctx_objets, "objet", f"pmi_obj_{id(ctx_objets)}")
        else:
            st.info("Pas de données d'objets disponibles.")


# ── Carte ─────────────────────────────────────────────────────────────────────

def tab_carte(
    locs:    pd.DataFrame,
    posts:   pd.DataFrame,
    acts:    pd.DataFrame,
    imps:    pd.DataFrame,
    acteurs: pd.DataFrame,
) -> None:
    section("Carte des localisations")

    if locs.empty or "latitude" not in locs.columns:
        st.info("Aucune donnée de localisation disponible.")
        return

    locs_w = locs.copy()

    locs_w["_lat_r"] = locs_w["latitude"].round(4)
    locs_w["_lon_r"] = locs_w["longitude"].round(4)

    extra_first = {
        c: (c, "first") for c in ["label", "city"] if c in locs_w.columns
    }
    agg = (
        locs_w.groupby(["_lat_r", "_lon_r"])
        .agg(
            n_posts  =("Id_anonym", "nunique"),
            latitude =("latitude",  "first"),
            longitude=("longitude", "first"),
            **extra_first,
        )
        .reset_index()
    )

    max_posts   = agg["n_posts"].max()
    agg["_size"] = 4 + 36 * (np.sqrt(agg["n_posts"]) / max(np.sqrt(max_posts), 1))
    name_col = "label" if "label" in agg.columns else ("city" if "city" in agg.columns else None)
    agg["_hover"] = (
        "<b>" + (agg[name_col].fillna("?") if name_col else "?") + "</b><br>"
        + agg["n_posts"].astype(str) + " posts"
    )

    fig = go.Figure(go.Scattermap(
        lat=agg["latitude"], lon=agg["longitude"],
        mode="markers",
        marker=dict(size=agg["_size"], color="#4169E1", opacity=0.75),
        text=agg["_hover"],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=agg["latitude"].median(), lon=agg["longitude"].median()),
            zoom=6,
        ),
        height=580,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Taille ∝ √(nb posts) · {len(agg)} lieux · "
        f"{int(agg['n_posts'].sum())} occurrences"
    )


# ── Export ────────────────────────────────────────────────────────────────────

def tab_export(ctx: dict, label: str) -> None:
    """
    Génère un ZIP avec tous les fichiers du dict ctx.
    Les clés deviennent les noms de fichiers dans le ZIP.
    """
    section("Export")

    # Mapping clé → nom de fichier lisible dans le ZIP
    FILE_NAMES = {
        "activites":         "activites.csv",
        "impacts_neg":       "impacts_negatifs.csv",
        "impacts_pos":       "impacts_positifs.csv",
        "impacts_neutre":    "impacts_neutres.csv",
        "acteurs_humain":    "acteurs_humains.csv",
        "acteurs_non_humain":"acteurs_non_humains.csv",
        "localisations":     "localisations.csv",
        "ctx_objets":        "objets.csv",
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, filename in FILE_NAMES.items():
            df = ctx.get(key, pd.DataFrame())
            if not df.empty:
                zf.writestr(filename, df.to_csv(index=False))
    buf.seek(0)

    st.download_button(
        "⬇️ Télécharger les CSV (.zip)",
        data=buf.getvalue(),
        file_name=f"{label.lower().replace(' ', '_')}_export.zip",
        mime="application/zip",
    )

# ══════════════════════════════════════════════════════════════════════════════
# Fonctions _agg : pour la page globale (fichiers agrégés n_posts)
# Les fichiers ont déjà été filtrés par filter_agg() dans globale.py.
# ══════════════════════════════════════════════════════════════════════════════

from data.filters import compute_pmi_agg, top_n_agg


# ── Objets (agrégé) ───────────────────────────────────────────────────────────

def tab_objets_agg(ctx_f: dict) -> None:
    section("Distribution des objets détectés")

    df = ctx_f.get("obj_saison", pd.DataFrame())
    if df.empty or "objet" not in df.columns:
        st.info("Aucune donnée d'objets disponible.")
        return

    # Exclure le sentinel __all__
    df = df[df["objet"] != "__all__"]

    obj_counts = (
        df.groupby("objet")["n_posts"].sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"objet": "Objet", "n_posts": "Occurrences"})
    )

    df_pmi = ctx_f.get("obj_saison", pd.DataFrame())
    df_pmi = df_pmi[df_pmi["objet"] != "__all__"] if not df_pmi.empty else df_pmi

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(obj_counts, x="Occurrences", y="Objet", orientation="h",
                     color="Occurrences",
                     color_continuous_scale=["#c8ddf5", "#0d1b2a"])
        fig.update_layout(height=max(300, len(obj_counts) * 22 + 60),
                          plot_bgcolor=_BG, paper_bgcolor=_BG,
                          coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"), margin=_MARGIN)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if not df_pmi.empty and "saison" in df_pmi.columns:
            st.markdown("**Objets × Saison — PMI**")
            pmi_df = compute_pmi_agg(df_pmi, "objet", "saison")
            if not pmi_df.empty:
                mat = pmi_df.pivot(index="objet", columns="saison", values="pmi").fillna(0)
                fig2 = px.imshow(mat, text_auto=".2f", aspect="auto",
                                 color_continuous_scale="RdBu",
                                 color_continuous_midpoint=0, zmin=-3, zmax=3)
                fig2.update_layout(height=max(350, len(mat) * 22 + 80), margin=_MARGIN)
                st.plotly_chart(fig2, use_container_width=True)


# ── Activités & Impacts (agrégé) ──────────────────────────────────────────────

def tab_act_imp_agg(ctx_f: dict) -> None:
    section("Activités & Impacts")

    act_df  = ctx_f.get("act_saison",  pd.DataFrame())
    imp_df  = ctx_f.get("imp_saison",  pd.DataFrame())

    # Séparation impacts par polarité
    def _imp(pattern: str) -> pd.DataFrame:
        if imp_df.empty or "Type_impact" not in imp_df.columns:
            return pd.DataFrame()
        return imp_df[imp_df["Type_impact"].str.lower().str.contains(pattern, na=False)]

    imps_neg = _imp(r"n[eé]gatif")
    imps_pos = _imp("positif")
    imps_neu = _imp("neutre")

    n_act = max(5, act_df["label_merged_act"].dropna().nunique()) if not act_df.empty and "label_merged_act" in act_df.columns else 5
    n_imp = max(5, imp_df["label_merged_imp"].dropna().nunique()) if not imp_df.empty and "label_merged_imp" in imp_df.columns else 5

    col1, col2 = st.columns(2)
    with col1:
        top_n_act = st.slider("Top N activités", 1, n_act, min(20, n_act), key="sl_act_agg")
    with col2:
        top_n_imp = st.slider("Top N impacts", 1, n_imp, min(10, n_imp), key="sl_imp_agg")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**🔵 Activités**")
        if not act_df.empty:
            df_top = top_n_agg(act_df, "label_merged_act", top_n_act)
            fig = px.bar(df_top, x="count", y="label", orientation="h",
                         color="count", color_continuous_scale=["#c8d8f5", "#4169E1"])
            fig.update_layout(height=max(300, len(df_top) * 22 + 40),
                              plot_bgcolor=_BG, paper_bgcolor=_BG,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"), margin=_MARGIN)
            st.plotly_chart(fig, use_container_width=True)

        # Camembert tonalité
        if not imp_df.empty:
            st.markdown("**Répartition des impacts**")
            tone_data = [
                {"Tonalité": "négatif", "Nb": int(imps_neg["n_posts"].sum()) if not imps_neg.empty else 0},
                {"Tonalité": "positif", "Nb": int(imps_pos["n_posts"].sum()) if not imps_pos.empty else 0},
                {"Tonalité": "neutre",  "Nb": int(imps_neu["n_posts"].sum()) if not imps_neu.empty else 0},
            ]
            tone = pd.DataFrame([t for t in tone_data if t["Nb"] > 0])
            if not tone.empty:
                fig2 = px.pie(tone, names="Tonalité", values="Nb", hole=0.45,
                              color="Tonalité",
                              color_discrete_map={"positif": "#3CB371",
                                                  "négatif": "#DC143C",
                                                  "neutre":  "#9e9e9e"})
                fig2.update_layout(height=220, margin=_MARGIN)
                st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("**🔴 Impacts négatifs**")
        if not imps_neg.empty:
            df_neg = top_n_agg(imps_neg, "label_merged_imp", top_n_imp)
            fig = px.bar(df_neg, x="count", y="label", orientation="h",
                         color="count", color_continuous_scale=["#fcd0d0", "#DC143C"])
            fig.update_layout(height=max(250, len(df_neg) * 22 + 40),
                              plot_bgcolor=_BG, paper_bgcolor=_BG,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"), margin=_MARGIN)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**🟢 Impacts positifs**")
        if not imps_pos.empty:
            df_pos = top_n_agg(imps_pos, "label_merged_imp", top_n_imp)
            fig = px.bar(df_pos, x="count", y="label", orientation="h",
                         color="count", color_continuous_scale=["#d4f5e2", "#3CB371"])
            fig.update_layout(height=max(250, len(df_pos) * 22 + 40),
                              plot_bgcolor=_BG, paper_bgcolor=_BG,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"), margin=_MARGIN)
            st.plotly_chart(fig, use_container_width=True)

        if not imps_neu.empty:
            st.markdown("**⚪ Impacts neutres**")
            df_neu = top_n_agg(imps_neu, "label_merged_imp", top_n_imp)
            fig = px.bar(df_neu, x="count", y="label", orientation="h",
                         color="count", color_continuous_scale=["#e8e8e8", "#9e9e9e"])
            fig.update_layout(height=max(200, len(df_neu) * 22 + 40),
                              plot_bgcolor=_BG, paper_bgcolor=_BG,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"), margin=_MARGIN)
            st.plotly_chart(fig, use_container_width=True)


# ── Acteurs (agrégé) ──────────────────────────────────────────────────────────

def tab_acteurs_agg(ctx_f: dict) -> None:
    section("Acteurs identifiés")

    # Fichiers déjà séparés humain / non-humain
    actrs_hum = ctx_f.get("actor_hum_saison", pd.DataFrame())
    actrs_nh  = ctx_f.get("actor_nh_saison",  pd.DataFrame())

    # DataFrame combiné pour le top-N global
    actor_df = pd.concat(
        [df for df in [actrs_hum, actrs_nh] if not df.empty],
        ignore_index=True,
    ) if (not actrs_hum.empty or not actrs_nh.empty) else pd.DataFrame()

    if actor_df.empty or "label_merged_actor" not in actor_df.columns:
        st.info("Aucune donnée d'acteurs.")
        return

    c1, c2 = st.columns([3, 2])

    with c1:
        df_top = top_n_agg(actor_df, "label_merged_actor", TOP_N)
        hum_labels = set(
            actrs_hum["label_merged_actor"].str.lower().dropna().unique()
            if not actrs_hum.empty else []
        )
        bar_colors = [
            _TYPE_COLORS["humain"] if lbl in hum_labels else _TYPE_COLORS["non humain"]
            for lbl in df_top["label"]
        ]
        fig = go.Figure(go.Bar(x=df_top["count"], y=df_top["label"],
                               orientation="h", marker_color=bar_colors))
        fig.update_layout(height=520, plot_bgcolor=_BG, paper_bgcolor=_BG,
                          showlegend=False, yaxis=dict(autorange="reversed"),
                          margin=_MARGIN)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tc = pd.DataFrame([
            {"Type": "humain",     "Nb": int(actrs_hum["n_posts"].sum()) if not actrs_hum.empty else 0},
            {"Type": "non humain", "Nb": int(actrs_nh["n_posts"].sum())  if not actrs_nh.empty  else 0},
        ])
        fig2 = px.pie(tc, names="Type", values="Nb", color="Type",
                      color_discrete_map=_TYPE_COLORS, hole=0.5,
                      title="Humain vs non-humain")
        fig2.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ── PMI (agrégé) ──────────────────────────────────────────────────────────────

def tab_pmi_agg(ctx_f: dict) -> None:
    section("Associations — PMI")

    mode = st.radio("Contexte", ["Saison", "Façade maritime"],
                    horizontal=True, key="pmi_mode_agg")
    suffix = "saison" if mode == "Saison" else "facade"
    ctx_col = "saison" if suffix == "saison" else "meta_facade"

    sub_tabs = st.tabs([
        "🔵 Activités", "🔴 Impacts négatifs", "🟢 Impacts positifs",
        "🟣 Acteurs", "📦 Objets",
    ])

    def _show(df: pd.DataFrame, label_col: str, tab_key: str) -> None:
        if df.empty or label_col not in df.columns or ctx_col not in df.columns:
            st.info("Pas assez de données.")
            return
        pmi_df = compute_pmi_agg(df, label_col, ctx_col)
        if pmi_df.empty:
            st.info("Pas assez de données.")
            return
        mat = pmi_df.pivot(index=label_col, columns=ctx_col, values="pmi").fillna(0)
        top_ents = (
            df.groupby(label_col)["n_posts"].sum()
            .nlargest(30).index
        )
        mat = mat[mat.index.isin(top_ents)]
        fig = px.imshow(mat, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0, zmin=-3, zmax=3)
        fig.update_layout(height=max(400, len(mat) * 18 + 80), margin=_MARGIN)
        st.plotly_chart(fig, use_container_width=True, key=tab_key)

    imp_df = ctx_f.get(f"imp_{suffix}", pd.DataFrame())

    def _imp_pol(pattern: str) -> pd.DataFrame:
        if imp_df.empty or "Type_impact" not in imp_df.columns:
            return pd.DataFrame()
        return imp_df[imp_df["Type_impact"].str.lower().str.contains(pattern, na=False)]

    with sub_tabs[0]:
        _show(ctx_f.get(f"act_{suffix}", pd.DataFrame()),
              "label_merged_act", f"pmi_act_{suffix}")
    with sub_tabs[1]:
        _show(_imp_pol(r"n[eé]gatif"), "label_merged_imp", f"pmi_neg_{suffix}")
    with sub_tabs[2]:
        _show(_imp_pol("positif"),    "label_merged_imp", f"pmi_pos_{suffix}")
    with sub_tabs[3]:
        # Combiner humain + non-humain pour la PMI acteurs
        actor_hum = ctx_f.get(f"actor_hum_{suffix}", pd.DataFrame())
        actor_nh  = ctx_f.get(f"actor_nh_{suffix}",  pd.DataFrame())
        actor_combined = pd.concat(
            [df for df in [actor_hum, actor_nh] if not df.empty],
            ignore_index=True,
        ) if (not actor_hum.empty or not actor_nh.empty) else pd.DataFrame()
        _show(actor_combined, "label_merged_actor", f"pmi_actor_{suffix}")
    with sub_tabs[4]:
        obj_df = ctx_f.get(f"obj_{suffix}", pd.DataFrame())
        obj_df = obj_df[obj_df["objet"] != "__all__"] if not obj_df.empty else obj_df
        _show(obj_df, "objet", f"pmi_obj_{suffix}")


# ── Carte (agrégé) ────────────────────────────────────────────────────────────

def tab_carte_agg(
    ctx_f: dict,
    sel_saisons: list[str] | None = None,
    sel_facades: list[str] | None = None,
) -> None:
    """
    Choisit le fichier carte selon les filtres actifs :
    - filtre façade actif  → carte_facade  (colonne meta_facade)
    - filtre saison actif ou aucun filtre → carte_saison (colonne saison)
    Si les deux sont actifs, priorité à facade (plus discriminant).
    Le filtre objet est déjà appliqué en amont par filter_agg().
    """
    section("Carte des localisations")

    # Choix du fichier selon filtres actifs
    if sel_facades:
        df = ctx_f.get("carte_facade", pd.DataFrame())
    else:
        df = ctx_f.get("carte_saison", pd.DataFrame())

    if df.empty or "latitude" not in df.columns:
        st.info("Aucune donnée de localisation disponible.")
        return

    # Agréger par lieu (somme n_posts, toutes saisons/facades confondues)
    loc_extra = [c for c in ["label", "city"] if c in df.columns]
    agg = (
        df.groupby(["latitude", "longitude"] + loc_extra, dropna=False)["n_posts"]
        .sum()
        .reset_index()
    )

    max_posts    = agg["n_posts"].max()
    agg["_size"] = 4 + 36 * (np.sqrt(agg["n_posts"]) / max(np.sqrt(max_posts), 1))
    name_col     = "label" if "label" in agg.columns else ("city" if "city" in agg.columns else None)
    agg["_hover"] = (
        "<b>" + (agg[name_col].fillna("?") if name_col else "?") + "</b><br>"
        + agg["n_posts"].astype(str) + " posts"
    )

    fig = go.Figure(go.Scattermap(
        lat=agg["latitude"], lon=agg["longitude"], mode="markers",
        marker=dict(size=agg["_size"], color="#4169E1", opacity=0.75),
        text=agg["_hover"], hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        map=dict(style="carto-positron",
                 center=dict(lat=agg["latitude"].median(),
                             lon=agg["longitude"].median()),
                 zoom=5),
        height=580, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Taille ∝ √(nb posts) · {len(agg)} lieux · {int(agg['n_posts'].sum())} occurrences")