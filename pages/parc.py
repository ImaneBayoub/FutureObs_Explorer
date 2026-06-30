"""
FUTURE-Obs — Page exploration d'un parc naturel marin
======================================================
Onglets :
  UMAP        : projection sémantique
  Carte       : localisation des posts
  Objets      : distribution des objets détectés
  Activités   : top activités
  Impacts     : impacts négatifs / positifs / neutres
  Acteurs     : acteurs humains / non-humains
  Graph       : export .gexf filtré par fréquence
  Export CSV  : téléchargement des tables filtrées
"""

import io
import zipfile
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import FACADES_VALIDES, SAISONS, ZONES
from data.filters import filter_by_freq, filter_parc, top_n
from data.loader import load_stats, load_umap_html, load_zone
from ui.layout import header, section
from ui.tabs_shared import tab_carte, tab_overview_umap

_BG     = "#fafbfd"
_MARGIN = dict(l=10, r=10, t=10, b=10)
_TYPE_COLORS = {"humain": "#9370DB", "non humain": "#20B2AA"}

_UMAP_PARC_PATHS: dict[str, str] = {
    "mediterranee_sites": "umap_dashboard_mediterranee_sites.html",
}


def _safe_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """pd.concat sécurisé : retourne un DataFrame vide si la liste est vide
    ou ne contient aucun DataFrame non-vide."""
    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ── Sidebar filtres ───────────────────────────────────────────────────────────

def _render_sidebar(ctx: dict, parc_slug: str) -> dict:
    """Affiche les 7 filtres et retourne un dict des sélections.

    Les options proposées sont strictement limitées aux valeurs présentes
    dans le corpus du parc chargé (ctx) — aucune valeur globale ou
    provenant d'un autre parc n'est proposée.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtres")

    acts_all = ctx.get("activites", pd.DataFrame())
    imps_all = _safe_concat(
        [df for k, df in ctx.items() if k.startswith("impacts_")]
    )
    actrs_all = _safe_concat(
        [df for k, df in ctx.items() if k.startswith("acteurs_")]
    )
    locs   = ctx.get("localisations", pd.DataFrame())
    obj_df = ctx.get("ctx_objets", pd.DataFrame())

    def _opts(df: pd.DataFrame, col: str) -> list[str]:
        """Retourne les valeurs uniques réellement présentes dans df[col],
        castées en str, en minuscules, triées, sans NaN ni vides."""
        if df is None or df.empty or col not in df.columns:
            return []
        vals = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        vals = vals[vals != ""]
        return sorted(vals.unique().tolist())

    # Saison — on ne propose que les saisons réellement présentes dans le corpus du parc
    saison_cols = []
    for df in (acts_all, imps_all, actrs_all):
        if not df.empty and "saison" in df.columns:
            saison_cols.append(df["saison"])
    saisons_presentes = (
        sorted(pd.concat(saison_cols).dropna().astype(str).str.lower().unique().tolist())
        if saison_cols else []
    )
    saisons_opts = [s for s in SAISONS if s.lower() in saisons_presentes] if saisons_presentes else SAISONS

    sel_saisons = st.sidebar.multiselect(
        "Saison", saisons_opts, default=[], key=f"p_saison_{parc_slug}",
        placeholder="toutes",
    )

    # Façade — idem, restreint aux façades réellement présentes dans le corpus du parc
    facade_cols = []
    for df in (acts_all, imps_all, actrs_all, locs):
        if not df.empty and "facade" in df.columns:
            facade_cols.append(df["facade"])
    facades_presentes = (
        sorted(pd.concat(facade_cols).dropna().astype(str).str.lower().unique().tolist())
        if facade_cols else []
    )
    facades_opts = [f for f in FACADES_VALIDES if f.lower() in facades_presentes] if facades_presentes else FACADES_VALIDES

    sel_facades = st.sidebar.multiselect(
        "Façade", facades_opts, default=[], key=f"p_facade_{parc_slug}",
        placeholder="toutes",
    )

    # Objet — uniquement les objets présents dans ce parc
    obj_opts = _opts(obj_df, "objet")
    sel_objets = st.sidebar.multiselect(
        "Objet", obj_opts, default=[], key=f"p_objet_{parc_slug}",
        placeholder="tous",
    ) if obj_opts else []

    # Activité — uniquement les activités présentes dans ce parc
    act_opts = _opts(acts_all, "label_merged_act")
    sel_acts = st.sidebar.multiselect(
        "Activité", act_opts, default=[], key=f"p_act_{parc_slug}",
        placeholder="toutes",
    ) if act_opts else []

    # Impact — uniquement les impacts présents dans ce parc
    imp_opts = _opts(imps_all, "label_merged_imp")
    sel_imps = st.sidebar.multiselect(
        "Impact", imp_opts, default=[], key=f"p_imp_{parc_slug}",
        placeholder="tous",
    ) if imp_opts else []

    # Acteur — uniquement les acteurs présents dans ce parc
    actr_opts = _opts(actrs_all, "label_merged_actor")
    sel_actrs = st.sidebar.multiselect(
        "Acteur", actr_opts, default=[], key=f"p_actr_{parc_slug}",
        placeholder="tous",
    ) if actr_opts else []

    # Ville — uniquement les villes présentes dans ce parc
    ville_col = "label" if not locs.empty and "label" in locs.columns else (
        "city" if not locs.empty and "city" in locs.columns else None
    )
    ville_opts = _opts(locs, ville_col) if ville_col else []
    sel_villes = st.sidebar.multiselect(
        "Ville", ville_opts, default=[], key=f"p_ville_{parc_slug}",
        placeholder="toutes",
    ) if ville_opts else []

    return {
        "saisons":   sel_saisons or None,
        "facades":   sel_facades or None,
        "objets":    sel_objets  or None,
        "activites": sel_acts    or None,
        "impacts":   sel_imps    or None,
        "acteurs":   sel_actrs   or None,
        "villes":    sel_villes  or None,
    }


# ── Onglet Objets ─────────────────────────────────────────────────────────────

def _tab_objets(ctx_f: dict) -> None:
    section("Distribution des objets détectés")
    obj_df = ctx_f.get("ctx_objets", pd.DataFrame())
    if obj_df.empty or "objet" not in obj_df.columns:
        st.info("Aucune donnée d'objets disponible.")
        return
    counts = obj_df["objet"].dropna().value_counts().reset_index()
    counts.columns = ["Objet", "Occurrences"]
    fig = px.bar(counts, x="Occurrences", y="Objet", orientation="h",
                 color="Occurrences", color_continuous_scale=["#c8ddf5", "#0d1b2a"])
    fig.update_layout(height=max(300, len(counts) * 22 + 60),
                      plot_bgcolor=_BG, paper_bgcolor=_BG,
                      coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed"), margin=_MARGIN)
    st.plotly_chart(fig, width='stretch')


# ── Onglet Activités ──────────────────────────────────────────────────────────

def _tab_activites(ctx_f: dict) -> None:
    section("Activités")
    acts = ctx_f.get("activites", pd.DataFrame())
    if acts.empty or "label_merged_act" not in acts.columns:
        st.info("Aucune donnée d'activités.")
        return
    n = max(5, acts["label_merged_act"].dropna().nunique())
    top_n_val = st.slider("Top N", 1, n, min(25, n), key="sl_act_parc")
    df_top = top_n(acts, "label_merged_act", top_n_val)
    fig = px.bar(df_top, x="count", y="label", orientation="h",
                 color="count", color_continuous_scale=["#c8d8f5", "#4169E1"])
    fig.update_layout(height=max(300, len(df_top) * 22 + 40),
                      plot_bgcolor=_BG, paper_bgcolor=_BG,
                      coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed"), margin=_MARGIN)
    st.plotly_chart(fig, width='stretch')


# ── Onglet Impacts ────────────────────────────────────────────────────────────

def _tab_impacts(ctx_f: dict) -> None:
    section("Impacts")
    imp_neg = ctx_f.get("impacts_neg",    pd.DataFrame())
    imp_pos = ctx_f.get("impacts_pos",    pd.DataFrame())
    imp_neu = ctx_f.get("impacts_neutre", pd.DataFrame())

    imps_all = _safe_concat([imp_neg, imp_pos, imp_neu])
    if imps_all.empty:
        st.info("Aucune donnée d'impacts.")
        return

    n = max(5, imps_all["label_merged_imp"].dropna().nunique())
    top_n_val = st.slider("Top N", 1, n, min(15, n), key="sl_imp_parc")

    # Camembert tonalité
    tone_data = [
        {"Tonalité": "négatif", "Nb": len(imp_neg)},
        {"Tonalité": "positif", "Nb": len(imp_pos)},
        {"Tonalité": "neutre",  "Nb": len(imp_neu)},
    ]
    tone = pd.DataFrame([t for t in tone_data if t["Nb"] > 0])
    if not tone.empty:
        fig_pie = px.pie(tone, names="Tonalité", values="Nb", hole=0.45,
                         color="Tonalité",
                         color_discrete_map={"positif": "#3CB371",
                                             "négatif": "#DC143C",
                                             "neutre":  "#9e9e9e"})
        fig_pie.update_layout(height=220, margin=_MARGIN)
        st.plotly_chart(fig_pie, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Négatifs**")
        if not imp_neg.empty:
            df_top = top_n(imp_neg, "label_merged_imp", top_n_val)
            fig = px.bar(df_top, x="count", y="label", orientation="h",
                         color="count", color_continuous_scale=["#fcd0d0", "#DC143C"])
            fig.update_layout(height=max(250, len(df_top) * 22 + 40),
                              plot_bgcolor=_BG, paper_bgcolor=_BG,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"), margin=_MARGIN)
            st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown("**Positifs**")
        if not imp_pos.empty:
            df_top = top_n(imp_pos, "label_merged_imp", top_n_val)
            fig = px.bar(df_top, x="count", y="label", orientation="h",
                         color="count", color_continuous_scale=["#d4f5e2", "#3CB371"])
            fig.update_layout(height=max(250, len(df_top) * 22 + 40),
                              plot_bgcolor=_BG, paper_bgcolor=_BG,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"), margin=_MARGIN)
            st.plotly_chart(fig, width='stretch')


# ── Onglet Acteurs ────────────────────────────────────────────────────────────

def _tab_acteurs(ctx_f: dict) -> None:
    section("Acteurs")
    actrs_hum = ctx_f.get("acteurs_humain",     pd.DataFrame())
    actrs_nh  = ctx_f.get("acteurs_non_humain", pd.DataFrame())

    actrs_all = _safe_concat([actrs_hum, actrs_nh])
    if actrs_all.empty:
        st.info("Aucune donnée d'acteurs.")
        return

    c1, c2 = st.columns([3, 2])
    with c1:
        df_top = top_n(actrs_all, "label_merged_actor", 25)
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
        st.plotly_chart(fig, width='stretch')
    with c2:
        tc = pd.DataFrame([
            {"Type": "humain",     "Nb": len(actrs_hum)},
            {"Type": "non humain", "Nb": len(actrs_nh)},
        ])
        fig2 = px.pie(tc, names="Type", values="Nb", color="Type",
                      color_discrete_map=_TYPE_COLORS, hole=0.5,
                      title="Humain vs non-humain")
        fig2.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, width='stretch')


# ── Onglet Graph ──────────────────────────────────────────────────────────────

def _tab_graph(ctx_f: dict) -> None:
    section("Export graphe (.gexf)")
    st.caption("Les nœuds sont les entités (activités, impacts, acteurs). "
               "Les arêtes relient deux entités partageant le même Activite_ID.")

    min_freq = st.slider("Fréquence minimale des entités", 1, 50, 2,
                         key="graph_min_freq")

    acts  = ctx_f.get("activites", pd.DataFrame())
    imps  = _safe_concat([df for k, df in ctx_f.items() if k.startswith("impacts_")])
    actrs = _safe_concat([df for k, df in ctx_f.items() if k.startswith("acteurs_")])

    # Filtrage par fréquence
    acts  = filter_by_freq(acts,  "label_merged_act",   min_freq)
    imps  = filter_by_freq(imps,  "label_merged_imp",   min_freq)
    actrs = filter_by_freq(actrs, "label_merged_actor", min_freq)

    if acts.empty and imps.empty and actrs.empty:
        st.info("Aucune entité après filtrage par fréquence.")
        return

    # Construction du graphe
    G = nx.Graph()

    # Nœuds
    for label, type_node in [
        (acts,  "activite"),
        (imps,  "impact"),
        (actrs, "acteur"),
    ]:
        col = {"activite": "label_merged_act",
               "impact":   "label_merged_imp",
               "acteur":   "label_merged_actor"}[type_node]
        if label.empty or col not in label.columns:
            continue
        for entity in label[col].dropna().unique():
            freq = int((label[col] == entity).sum())
            if not G.has_node(entity):
                G.add_node(entity, type=type_node, weight=freq)

    # Arêtes via Activite_ID partagé
    all_entities = _safe_concat([
        acts[["Activite_ID", "label_merged_act"]].rename(columns={"label_merged_act": "entity"})
            if not acts.empty and "Activite_ID" in acts.columns else pd.DataFrame(),
        imps[["Activite_ID", "label_merged_imp"]].rename(columns={"label_merged_imp": "entity"})
            if not imps.empty and "Activite_ID" in imps.columns else pd.DataFrame(),
        actrs[["Activite_ID", "label_merged_actor"]].rename(columns={"label_merged_actor": "entity"})
            if not actrs.empty and "Activite_ID" in actrs.columns else pd.DataFrame(),
    ]).dropna()

    if not all_entities.empty:
        grouped = all_entities.groupby("Activite_ID")["entity"].apply(list)
        for entities in grouped:
            entities = [e for e in entities if G.has_node(e)]
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    u, v = entities[i], entities[j]
                    if G.has_edge(u, v):
                        G[u][v]["weight"] = G[u][v].get("weight", 1) + 1
                    else:
                        G.add_edge(u, v, weight=1)

    st.markdown(f"**{G.number_of_nodes()} nœuds · {G.number_of_edges()} arêtes**")

    # ── Aperçu visuel du graphe ───────────────────────────────────────────────
    if G.number_of_nodes() > 0:
        # Layout spring pour les petits graphes, sinon random
        if G.number_of_nodes() <= 300:
            pos = nx.spring_layout(G, seed=42, k=0.5)
        else:
            pos = nx.random_layout(G, seed=42)

        # Couleur par type de nœud
        NODE_COLORS = {"activite": "#4169E1", "impact": "#DC143C", "acteur": "#9370DB"}

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_colors = [NODE_COLORS.get(G.nodes[n].get("type", "activite"), "#4169E1")
                       for n in G.nodes()]
        node_sizes  = [max(6, min(30, G.nodes[n].get("weight", 1) * 2))
                       for n in G.nodes()]
        node_text   = [f"{n}<br>type: {G.nodes[n].get('type','')}<br>"
                       f"freq: {G.nodes[n].get('weight',1)}"
                       for n in G.nodes()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="#cccccc"), hoverinfo="none",
        ))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(size=node_sizes, color=node_colors, opacity=0.85,
                        line=dict(width=0.5, color="white")),
            text=node_text, hovertemplate="%{text}<extra></extra>",
        ))
        fig.update_layout(
            height=500, showlegend=False,
            plot_bgcolor=_BG, paper_bgcolor=_BG,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Activités · Impacts · Acteurs — taille ∝ fréquence")

    # Export GEXF
    buf = io.BytesIO()
    nx.write_gexf(G, buf)
    buf.seek(0)
    st.download_button(
        "Télécharger le graphe (.gexf)",
        data=buf.getvalue(),
        file_name="graph_futureobs.gexf",
        mime="application/gexf+xml",
    )


# ── Onglet Export CSV ─────────────────────────────────────────────────────────

def _tab_export_csv(ctx_f: dict, zone_label: str) -> None:
    section("Export CSV")

    min_freq = st.slider("Fréquence minimale des entités", 1, 50, 1,
                         key="export_min_freq")

    tables = {
        "activites":          ("Activités",    "label_merged_act"),
        "impacts_neg":        ("Impacts −",    "label_merged_imp"),
        "impacts_pos":        ("Impacts +",    "label_merged_imp"),
        "impacts_neutre":     ("Impacts ∅",    "label_merged_imp"),
        "acteurs_humain":     ("Acteurs H",    "label_merged_actor"),
        "acteurs_non_humain": ("Acteurs NH",   "label_merged_actor"),
        "localisations":      ("Localisations", None),
        "ctx_objets":         ("Objets",        "objet"),
    }

    sel_tables = st.multiselect(
        "Tables à télécharger",
        options=list(tables.keys()),
        default=list(tables.keys()),
        format_func=lambda k: tables[k][0],
        key="export_sel_tables",
    )

    # ── Aperçu des tables ────────────────────────────────────────────────────
    for key in sel_tables:
        label_name, label_col = tables[key]
        df = ctx_f.get(key, pd.DataFrame())
        if df.empty:
            continue
        df_preview = df.copy()
        if label_col and min_freq > 1:
            df_preview = filter_by_freq(df_preview, label_col, min_freq)
        if df_preview.empty:
            continue
        with st.expander(f"**{label_name}** — {len(df_preview):,} lignes", expanded=False):
            st.dataframe(df_preview.head(20), width='stretch', hide_index=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in sel_tables:
            label_name, label_col = tables[key]
            df = ctx_f.get(key, pd.DataFrame())
            if df.empty:
                continue
            if label_col and min_freq > 1:
                df = filter_by_freq(df, label_col, min_freq)
            if not df.empty:
                zf.writestr(f"{key}.csv", df.to_csv(index=False))
    buf.seek(0)

    st.download_button(
        "⬇️ Télécharger les CSV (.zip)",
        data=buf.getvalue(),
        file_name=f"{zone_label.lower().replace(' ', '_')}_export.zip",
        mime="application/zip",
    )


# ── Page principale ───────────────────────────────────────────────────────────

def page_parc(zone_label: str, token: str) -> None:
    parc_slug = ZONES[zone_label]

    with st.spinner("Chargement des données du parc…"):
        ctx = load_zone(parc_slug, token)

    stats = load_stats(parc_slug)

    # Filtres sidebar
    sel = _render_sidebar(ctx, parc_slug)

    # Application du filtrage cumulatif
    ctx_f = filter_parc(
        ctx,
        saisons=sel["saisons"],
        facades=sel["facades"],
        objets=sel["objets"],
        activites=sel["activites"],
        impacts=sel["impacts"],
        acteurs=sel["acteurs"],
        villes=sel["villes"],
    )

    # Métriques pour le header (après filtrage)
    acts_df   = ctx_f.get("activites", pd.DataFrame())
    locs_df   = ctx_f.get("localisations", pd.DataFrame())
    imps_all  = _safe_concat(
        [df for k, df in ctx_f.items() if k.startswith("impacts_")]
    )
    actrs_all = _safe_concat(
        [df for k, df in ctx_f.items() if k.startswith("acteurs_")]
    )

    # Posts uniques réellement présents après filtrage
    post_id_col = "Id_anonym"
    post_sources = [acts_df, imps_all, actrs_all, locs_df]
    post_id_series = [
        df[post_id_col] for df in post_sources
        if not df.empty and post_id_col in df.columns
    ]
    if post_id_series:
        post_ids = pd.concat(post_id_series, ignore_index=True).dropna()
        n_posts_filtre = post_ids.nunique()
    else:
        n_posts_filtre = stats.get("n_posts", 0)

    def _n_unique(df: pd.DataFrame, col: str) -> int:
        """Nombre d'entités uniques (label_merged_*) plutôt que de lignes."""
        if df.empty or col not in df.columns:
            return 0
        return df[col].dropna().nunique()

    n_activites = _n_unique(acts_df,  "label_merged_act")
    n_impacts   = _n_unique(imps_all, "label_merged_imp")
    n_acteurs   = _n_unique(actrs_all, "label_merged_actor")

    header(
        title=zone_label,
        subtitle="FUTURE-Obs · Parc naturel marin",
        stats={
            "Posts":         n_posts_filtre,
            "Activités":     n_activites,
            "Impacts":       n_impacts,
            "Acteurs":       n_acteurs,
            "Localisations": len(locs_df),
        },
    )

    tabs = st.tabs([
        "UMAP",
        "Carte",
        "Objets",
        "Activités",
        "Impacts",
        "Acteurs",
        "Graph",
        "Export CSV",
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

        # Déduction automatique du mode
        if parc_slug in _UMAP_PARC_PATHS:
            mode = "multi"
            label = "Comparaison des sites côtiers"
        else:
            mode = "parc"
            label = zone_label

        tab_overview_umap(
            umap_html=umap_html,
            label=label,
            mode=mode,
        )

    with tabs[1]:
        tab_carte(locs_df, pd.DataFrame(), acts_df, imps_all, actrs_all)

    with tabs[2]:
        _tab_objets(ctx_f)

    with tabs[3]:
        _tab_activites(ctx_f)

    with tabs[4]:
        _tab_impacts(ctx_f)

    with tabs[5]:
        _tab_acteurs(ctx_f)

    with tabs[6]:
        _tab_graph(ctx_f)

    with tabs[7]:
        _tab_export_csv(ctx_f, zone_label)