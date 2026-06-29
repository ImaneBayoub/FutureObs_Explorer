"""
FUTURE-Obs — Page comparaison des parcs
=========================================
Charge dynamiquement chaque parc via load_zone(),
applique des filtres globaux, et affiche :
  - UMAP multi-parcs
  - Carte combinée des parcs
  - Top objets (global tous parcs)
  - Top entités (Activités, Impacts, Acteurs) comparaison 2 à 2
  - Heatmaps PMI (Objet × Parc, Saison × Parc)
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import FACADES_VALIDES, SAISONS, ZONE_COLORS, ZONES
from data.filters import compute_pmi, filter_parc, top_n
from data.loader import load_umap_html, load_zone
from ui.layout import header, section
from ui.tabs_shared import tab_overview_umap

_BG = "#fafbfd"
_MARGIN = dict(l=10, r=10, t=10, b=10)

# Marge spécifique pour les heatmaps PMI (laisse de la place aux labels longs)
_MARGIN_HEATMAP = dict(l=160, r=40, t=50, b=40)

# Parcs à comparer : tous sauf "Méditerranée - sites côtiers"
_PARCS_A_COMPARER = [z for z in ZONES.keys() if z != "Méditerranée - sites côtiers"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar_cmp(parcs_raw: dict[str, dict]) -> dict:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtres globaux")

    if not parcs_raw:
        return {}

    # Concaténer toutes les données pour extraire les options de filtres
    all_acts = pd.concat(
        [v.get("activites", pd.DataFrame()) for v in parcs_raw.values()],
        ignore_index=True
    )
    all_imps = pd.concat(
        [v.get(k, pd.DataFrame()) for v in parcs_raw.values()
         for k in v if k.startswith("impacts_") and not v.get(k).empty],
        ignore_index=True
    )
    all_actrs = pd.concat(
        [v.get(k, pd.DataFrame()) for v in parcs_raw.values()
         for k in v if k.startswith("acteurs_") and not v.get(k).empty],
        ignore_index=True
    )
    all_objs = pd.concat(
        [v.get("ctx_objets", pd.DataFrame()) for v in parcs_raw.values()],
        ignore_index=True
    )

    def _opts(df, col):
        if df.empty or col not in df.columns:
            return []
        return sorted(df[col].dropna().str.lower().unique().tolist())

    sel_saisons = st.sidebar.multiselect(
        "Saison", SAISONS, default=[], placeholder="toutes", key="cmp_saison"
    )
    sel_facades = st.sidebar.multiselect(
        "Façade", FACADES_VALIDES, default=[], placeholder="toutes", key="cmp_facade"
    )

    obj_opts = _opts(all_objs, "objet")
    sel_objets = (
        st.sidebar.multiselect(
            "Objet", obj_opts, default=[], placeholder="tous", key="cmp_objet"
        )
        if obj_opts else []
    )

    act_opts = _opts(all_acts, "label_merged_act")
    sel_acts = (
        st.sidebar.multiselect(
            "Activité", act_opts, default=[], placeholder="toutes", key="cmp_act"
        )
        if act_opts else []
    )

    imp_opts = _opts(all_imps, "label_merged_imp")
    sel_imps = (
        st.sidebar.multiselect(
            "Impact", imp_opts, default=[], placeholder="tous", key="cmp_imp"
        )
        if imp_opts else []
    )

    actr_opts = _opts(all_actrs, "label_merged_actor")
    sel_actrs = (
        st.sidebar.multiselect(
            "Acteur", actr_opts, default=[], placeholder="tous", key="cmp_actr"
        )
        if actr_opts else []
    )

    return {
        "saisons":   sel_saisons or None,
        "facades":   sel_facades or None,
        "objets":    sel_objets  or None,
        "activites": sel_acts    or None,
        "impacts":   sel_imps    or None,
        "acteurs":   sel_actrs   or None,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar_chart(df_top: pd.DataFrame, color: str, n: int) -> None:
    fig = px.bar(
        df_top, x="count", y="label", orientation="h",
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        height=max(250, n * 22 + 40),
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        yaxis=dict(autorange="reversed"),
        margin=_MARGIN,
    )
    st.plotly_chart(fig, use_container_width=True)


def _col_entity(label: str, df: pd.DataFrame, label_col: str,
                color: str, n: int) -> None:
    """Affiche le titre puis le bar chart top-N pour un classement donné."""
    st.markdown(f"**{label}**")
    if df.empty or label_col not in df.columns:
        st.caption("(pas de données)")
        return
    df_top = top_n(df, label_col, n)
    _bar_chart(df_top, color, n)


def _tab_entities_compare(
    title: str,
    parcs_f: dict[str, dict],
    dict_key: str,
    label_col: str,
    color_global: str,
    selected: list[str],
) -> None:
    """Affiche 2 classements côte à côte, choisis par l'utilisateur."""
    section(title)

    # Options disponibles
    options = ["Tous parcs"] + list(selected)

    # ── Ligne de contrôles ────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        choice_a = st.selectbox(
            "Comparer", options, index=0,
            key=f"cmp_{dict_key}_a",
        )
    with c2:
        choice_b = st.selectbox(
            "avec", options, index=min(1, len(options) - 1),
            key=f"cmp_{dict_key}_b",
        )
    with c3:
        top_n_val = st.slider(
            "Top", min_value=5, max_value=50, value=20,
            key=f"cmp_{dict_key}_n",
        )

    # ── Résoudre le DataFrame pour chaque choix ──────────────────────────
    def _resolve(choice: str) -> pd.DataFrame:
        if choice == "Tous parcs":
            return pd.concat(
                [ctx_f.get(dict_key, pd.DataFrame())
                 for ctx_f in parcs_f.values()],
                ignore_index=True,
            )
        return parcs_f[choice].get(dict_key, pd.DataFrame())

    df_a = _resolve(choice_a)
    df_b = _resolve(choice_b)

    # ── Deux colonnes de résultat ─────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        is_global = choice_a == "Tous parcs"
        _col_entity(
            choice_a, df_a, label_col,
            color_global if is_global
            else ZONE_COLORS.get(ZONES[choice_a], "#4169E1"),
            top_n_val,
        )

    with col_right:
        is_global = choice_b == "Tous parcs"
        _col_entity(
            choice_b, df_b, label_col,
            color_global if is_global
            else ZONE_COLORS.get(ZONES[choice_b], "#4169E1"),
            top_n_val,
        )


# ── Onglets spécifiques comparaison ───────────────────────────────────────────

def _tab_carte_compare(parcs_f: dict[str, dict]) -> None:
    section("Localisations des posts par parc")
    locs_list = []
    for zone_label, ctx_f in parcs_f.items():
        df = ctx_f.get("localisations", pd.DataFrame()).copy()
        if not df.empty:
            df["parc"] = zone_label
            locs_list.append(df)

    if not locs_list:
        st.info("Aucune donnée de localisation disponible pour les parcs sélectionnés.")
        return

    all_locs = pd.concat(locs_list, ignore_index=True)

    lat_col = "lat" if "lat" in all_locs.columns else "latitude"
    lon_col = "lon" if "lon" in all_locs.columns else "longitude"

    all_locs = all_locs.dropna(subset=[lat_col, lon_col])

    if all_locs.empty:
        st.info("Aucune coordonnée valide (lat/lon) après nettoyage.")
        return

    fig = px.scatter_mapbox(
        all_locs,
        lat=lat_col,
        lon=lon_col,
        color="parc",
        hover_name="label" if "label" in all_locs.columns else None,
        zoom=5,
        height=600,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _tab_pmi_compare(parcs_f: dict[str, dict]) -> None:
    section("Analyse PMI croisée")

    # 1. PMI Objet × Parc
    st.markdown("#### PMI Objet × Parc")
    objs_list = []
    for zone_label, ctx_f in parcs_f.items():
        df = ctx_f.get("ctx_objets", pd.DataFrame())
        if not df.empty and "objet" in df.columns:
            df_c = df[["objet"]].copy()
            df_c["parc"] = zone_label
            objs_list.append(df_c)

    if objs_list:
        all_objs = pd.concat(objs_list, ignore_index=True)
        pmi_obj_parc = compute_pmi(all_objs, "objet", "parc")
        if not pmi_obj_parc.empty:
            mat_obj = pmi_obj_parc.pivot(
                index="objet", columns="parc", values="pmi"
            ).fillna(0)
            
            # Hauteur dynamique : 28px par ligne pour éviter l'écrasement
            n_rows = len(mat_obj)
            dynamic_height = max(400, n_rows * 28 + 100)
            
            fig1 = px.imshow(
                mat_obj, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu", color_continuous_midpoint=0,
                zmin=-3, zmax=3,
                title="Spécificité des objets par parc",
            )
            # Ajustement fin pour éviter le chevauchement du texte
            fig1.update_traces(
                textfont_size=11, 
                textfont_color="black",
                xgap=3,  # Espace horizontal entre les cellules
                ygap=2   # Espace vertical entre les cellules
            )
            fig1.update_layout(
                height=dynamic_height, 
                margin=_MARGIN_HEATMAP,
                yaxis=dict(automargin=True),  # Agrandit la marge si les labels sont longs
                xaxis=dict(automargin=True, tickangle=0)
            )
            st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Aucune donnée d'objet pour calculer cette PMI.")

    st.markdown("---")

    # 2. PMI Saison × Parc
    st.markdown("#### PMI Saison × Parc")
    sais_list = []
    for zone_label, ctx_f in parcs_f.items():
        df = ctx_f.get("ctx_objets", pd.DataFrame())
        if df.empty or "saison" not in df.columns:
            df = ctx_f.get("activites", pd.DataFrame())
        if not df.empty and "saison" in df.columns:
            df_c = df[["saison"]].copy()
            df_c["parc"] = zone_label
            sais_list.append(df_c)

    if sais_list:
        all_sais = pd.concat(sais_list, ignore_index=True)
        pmi_sais_parc = compute_pmi(all_sais, "saison", "parc")
        if not pmi_sais_parc.empty:
            mat_sais = pmi_sais_parc.pivot(
                index="saison", columns="parc", values="pmi"
            ).fillna(0)
            
            # Hauteur dynamique adaptée aux saisons (moins de lignes que les objets)
            n_rows = len(mat_sais)
            dynamic_height = max(250, n_rows * 50 + 100)
            
            fig2 = px.imshow(
                mat_sais, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu", color_continuous_midpoint=0,
                zmin=-3, zmax=3,
                title="Spécificité des saisons par parc",
            )
            fig2.update_traces(
                textfont_size=12, 
                textfont_color="black",
                xgap=3, 
                ygap=3
            )
            fig2.update_layout(
                height=dynamic_height, 
                margin=_MARGIN_HEATMAP,
                yaxis=dict(automargin=True),
                xaxis=dict(automargin=True, tickangle=0)
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Aucune donnée de saison pour calculer cette PMI.")


# ── Page principale ───────────────────────────────────────────────────────────

def page_comparaison(token: str) -> None:
    # 1. Parcs à comparer : tous sauf "Méditerranée - sites côtiers"
    selected = _PARCS_A_COMPARER

    # 2. Chargement des données brutes
    with st.spinner("Chargement des données brutes des parcs…"):
        parcs_raw: dict[str, dict] = {
            zone_label: load_zone(ZONES[zone_label], token)
            for zone_label in selected
        }

    # 3. Récupération des filtres via la sidebar
    filters = _render_sidebar_cmp(parcs_raw)

    # 4. Application des filtres à chaque parc
    with st.spinner("Application des filtres…"):
        parcs_f: dict[str, dict] = {
            zone_label: filter_parc(
                ctx,
                saisons=filters.get("saisons"),
                facades=filters.get("facades"),
                objets=filters.get("objets"),
                activites=filters.get("activites"),
                impacts=filters.get("impacts"),
                acteurs=filters.get("acteurs"),
                villes=None,
            )
            for zone_label, ctx in parcs_raw.items()
        }

    # 5. En-tête
    header(
        title="Comparaison des parcs",
        subtitle="FUTURE-Obs · Vue comparative",
        stats={"Parcs comparés": len(selected)},
    )

    # 6. Onglets
    tabs_cmp = st.tabs([
        "Overview UMAP",
        "Carte",
        "Objets",
        "Activités",
        "Impacts",
        "Acteurs",
        "PMI Croisée",
    ])

    # ── Tab 0: Overview UMAP ──────────────────────────────────────────────────
    with tabs_cmp[0]:
        try:
            umap_html = load_umap_html("umap_dashboard_all_parcs.html", token)
        except Exception:
            umap_html = ""

        tab_overview_umap(
            umap_html=umap_html,
            label="Comparaison des parcs marins",
            mode="multi",
        )
    # ── Tab 1: Carte ──────────────────────────────────────────────────────────
    with tabs_cmp[1]:
        _tab_carte_compare(parcs_f)

    # ── Tab 2: Objets ─────────────────────────────────────────────────────────
    with tabs_cmp[2]:
        _tab_entities_compare(
            title="Top objets",
            parcs_f=parcs_f,
            dict_key="ctx_objets",
            label_col="objet",
            color_global="#4169E1",
            selected=selected,
        )

    # ── Tab 3: Activités ──────────────────────────────────────────────────────
    with tabs_cmp[3]:
        _tab_entities_compare(
            title="Top activités",
            parcs_f=parcs_f,
            dict_key="activites",
            label_col="label_merged_act",
            color_global="#FF8C00",
            selected=selected,
        )

    # ── Tab 4: Impacts (négatifs puis positifs) ───────────────────────────────
    with tabs_cmp[4]:
        _tab_entities_compare(
            title="Impacts négatifs",
            parcs_f=parcs_f,
            dict_key="impacts_neg",
            label_col="label_merged_imp",
            color_global="#DC143C",
            selected=selected,
        )

        st.markdown("---")

        _tab_entities_compare(
            title="Impacts positifs",
            parcs_f=parcs_f,
            dict_key="impacts_pos",
            label_col="label_merged_imp",
            color_global="#3CB371",
            selected=selected,
        )

    # ── Tab 5: Acteurs (humains puis non-humains) ─────────────────────────────
    with tabs_cmp[5]:
        _tab_entities_compare(
            title="Acteurs humains",
            parcs_f=parcs_f,
            dict_key="acteurs_humain",
            label_col="label_merged_actor",
            color_global="#9370DB",
            selected=selected,
        )

        st.markdown("---")

        _tab_entities_compare(
            title="Acteurs non-humains",
            parcs_f=parcs_f,
            dict_key="acteurs_non_humain",
            label_col="label_merged_actor",
            color_global="#20B2AA",
            selected=selected,
        )

    # ── Tab 6: PMI ────────────────────────────────────────────────────────────
    with tabs_cmp[6]:
        _tab_pmi_compare(parcs_f)