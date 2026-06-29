"""
FUTURE-Obs — Filtres et calculs sur les données
=================================================
  - filter_agg()   : filtre les fichiers agrégés globaux (page globale)
  - filter_parc()  : filtre cumulatif pour les pages parc
  - top_n()        : fréquence descendante sur fichiers bruts
  - top_n_agg()    : classement depuis fichiers agrégés
  - filter_by_freq(): filtre par fréquence minimale
  - compute_pmi()  : matrice PMI depuis fichiers bruts
  - compute_pmi_agg() : matrice PMI depuis fichiers agrégés
"""

import numpy as np
import pandas as pd

from config import PMI_MIN_N, TOP_N


# ── Filtrage fichiers agrégés (global) ───────────────────────────────────────

def filter_agg(
    ctx: dict[str, pd.DataFrame],
    objets:  list[str] | None = None,
    saisons: list[str] | None = None,
    facades: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Filtre les DataFrames agrégés du global (inchangé)."""
    out: dict[str, pd.DataFrame] = {}
    OBJ_FILES_NO_SENTINEL = {"obj_saison", "obj_facade", "carte_saison", "carte_facade"}

    for key, df in ctx.items():
        if df.empty:
            out[key] = df
            continue
        filtered = df.copy()

        if "objet" in filtered.columns:
            if objets:
                filtered = filtered[filtered["objet"].isin([o.lower() for o in objets])]
            elif key not in OBJ_FILES_NO_SENTINEL:
                filtered = filtered[filtered["objet"] == "__all__"]

        if saisons and "saison" in filtered.columns:
            filtered = filtered[filtered["saison"].isin(saisons)]

        if facades and "meta_facade" in filtered.columns:
            filtered = filtered[filtered["meta_facade"].isin(facades)]

        out[key] = filtered
    return out


# ── Filtrage cumulatif pour les parcs ────────────────────────────────────────

def filter_parc(
    ctx: dict[str, pd.DataFrame],
    saisons:   list[str] | None = None,
    facades:   list[str] | None = None,
    objets:    list[str] | None = None,
    activites: list[str] | None = None,
    impacts:   list[str] | None = None,
    acteurs:   list[str] | None = None,
    villes:    list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Filtre cumulatif pour les fichiers bruts d'un parc.

    Logique :
    1. Chaque filtre produit un ensemble d'Id_anonym valides.
    2. L'intersection de tous ces ensembles donne les Id_anonym retenus.
    3. Tous les DataFrames sont filtrés sur ces Id_anonym.

    Colonnes attendues par fichier :
      activites.csv       : Id_anonym, label_merged_act, saison, meta_facade
      impacts_*.csv       : Id_anonym, label_merged_imp, saison, meta_facade
      acteurs_*.csv       : Id_anonym, label_merged_actor, saison, meta_facade
      localisations.csv   : Id_anonym, label, city
      ctx_objets.csv      : Id_anonym, objet
    """
    acts_all  = pd.concat(
        [df for k, df in ctx.items()
         if k == "activites" and not df.empty],
        ignore_index=True,
    )
    imps_all  = pd.concat(
        [df for k, df in ctx.items()
         if k.startswith("impacts_") and not df.empty],
        ignore_index=True,
    )
    actrs_all = pd.concat(
        [df for k, df in ctx.items()
         if k.startswith("acteurs_") and not df.empty],
        ignore_index=True,
    )
    locs   = ctx.get("localisations", pd.DataFrame())
    objets_df = ctx.get("ctx_objets", pd.DataFrame())

    # ── Calcul des Id_anonym valides par filtre ───────────────────────────────
    id_sets: list[set[str]] = []

    def _ids(df: pd.DataFrame) -> set[str]:
        return set(df["Id_anonym"].dropna().astype(str).unique())

    # Saison
    if saisons and not acts_all.empty and "saison" in acts_all.columns:
        id_sets.append(_ids(acts_all[acts_all["saison"].isin(saisons)]))

    # Façade
    if facades and not acts_all.empty and "meta_facade" in acts_all.columns:
        id_sets.append(_ids(acts_all[acts_all["meta_facade"].isin(facades)]))

    # Objet
    if objets and not objets_df.empty and "objet" in objets_df.columns:
        mask = objets_df["objet"].isin([o.lower() for o in objets])
        id_sets.append(_ids(objets_df[mask]))

    # Activité
    if activites and not acts_all.empty and "label_merged_act" in acts_all.columns:
        mask = acts_all["label_merged_act"].isin([a.lower() for a in activites])
        id_sets.append(_ids(acts_all[mask]))

    # Impact
    if impacts and not imps_all.empty and "label_merged_imp" in imps_all.columns:
        mask = imps_all["label_merged_imp"].isin([i.lower() for i in impacts])
        id_sets.append(_ids(imps_all[mask]))

    # Acteur
    if acteurs and not actrs_all.empty and "label_merged_actor" in actrs_all.columns:
        mask = actrs_all["label_merged_actor"].isin([a.lower() for a in acteurs])
        id_sets.append(_ids(actrs_all[mask]))

    # Ville
    if villes and not locs.empty:
        ville_col = "label" if "label" in locs.columns else ("city" if "city" in locs.columns else None)
        if ville_col:
            mask = locs[ville_col].isin(villes)
            id_sets.append(_ids(locs[mask]))

    # ── Intersection ─────────────────────────────────────────────────────────
    if id_sets:
        kept_ids = id_sets[0]
        for s in id_sets[1:]:
            kept_ids &= s
    else:
        kept_ids = None  # pas de filtre actif → tout garder

    # ── Application aux DataFrames ────────────────────────────────────────────
    out: dict[str, pd.DataFrame] = {}
    for key, df in ctx.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[key] = df
            continue

        if kept_ids is not None and "Id_anonym" in df.columns:
            out[key] = df[df["Id_anonym"].astype(str).isin(kept_ids)].copy()
        else:
            out[key] = df

    return out


# ── Filtre par fréquence minimale ─────────────────────────────────────────────

def filter_by_freq(
    df: pd.DataFrame,
    label_col: str,
    min_freq: int,
) -> pd.DataFrame:
    """
    Garde uniquement les entités apparaissant au moins min_freq fois.
    Utilisé pour l'export graph et CSV.
    """
    if df.empty or label_col not in df.columns or min_freq <= 1:
        return df
    counts = df[label_col].value_counts()
    valid  = counts[counts >= min_freq].index
    return df[df[label_col].isin(valid)]


# ── Top-N fichiers bruts ──────────────────────────────────────────────────────

def top_n(
    df: pd.DataFrame,
    label_col: str,
    n: int = TOP_N,
) -> pd.DataFrame:
    """Top-N depuis fichiers bruts (value_counts sur les lignes)."""
    if df.empty or label_col not in df.columns:
        return pd.DataFrame(columns=["label", "count"])
    counts = (
        df[label_col].dropna().str.lower().str.strip()
        .value_counts().head(n).reset_index()
    )
    counts.columns = ["label", "count"]
    return counts


# ── Top-N fichiers agrégés ────────────────────────────────────────────────────

def top_n_agg(
    df: pd.DataFrame,
    label_col: str,
    n: int = TOP_N,
) -> pd.DataFrame:
    """Top-N depuis fichiers agrégés (somme de n_posts par label)."""
    if df.empty or label_col not in df.columns or "n_posts" not in df.columns:
        return pd.DataFrame(columns=["label", "count"])
    result = (
        df.groupby(label_col)["n_posts"]
        .sum()
        .nlargest(n)
        .reset_index()
        .rename(columns={label_col: "label", "n_posts": "count"})
    )
    return result


# ── PMI fichiers bruts ────────────────────────────────────────────────────────

def compute_pmi(
    df: pd.DataFrame,
    label_col: str,
    context_col: str,
) -> pd.DataFrame:
    """PMI depuis fichiers bruts."""
    df = df[[label_col, context_col]].dropna().copy()
    df[label_col]   = df[label_col].str.lower().str.strip()
    df[context_col] = df[context_col].str.strip()
    df = df[(df[label_col] != "") & (df[context_col] != "")]

    if len(df) < PMI_MIN_N:
        return pd.DataFrame()

    N = len(df)
    freq  = df[label_col].value_counts()
    valid = freq[freq >= PMI_MIN_N].index
    df    = df[df[label_col].isin(valid)]
    if df.empty:
        return pd.DataFrame()

    p_ent = df[label_col].value_counts()   / N
    p_ctx = df[context_col].value_counts() / N
    joint = df.groupby([label_col, context_col]).size() / N

    rows = []
    for (ent, ctx), p_j in joint.items():
        p_e = p_ent.get(ent, 0)
        p_c = p_ctx.get(ctx, 0)
        if p_e > 0 and p_c > 0 and p_j > 0:
            rows.append({
                label_col:   ent,
                context_col: ctx,
                "pmi":       round(float(np.log2(p_j / (p_e * p_c))), 3),
                "count":     int(round(p_j * N)),
            })
    return pd.DataFrame(rows)


# ── PMI fichiers agrégés ──────────────────────────────────────────────────────

def compute_pmi_agg(
    df: pd.DataFrame,
    label_col: str,
    context_col: str,
) -> pd.DataFrame:
    """PMI depuis fichiers agrégés (n_posts comme poids)."""
    if df.empty or label_col not in df.columns or "n_posts" not in df.columns:
        return pd.DataFrame()
    if context_col not in df.columns:
        return pd.DataFrame()

    df = df[[label_col, context_col, "n_posts"]].dropna().copy()
    df[label_col]   = df[label_col].str.lower().str.strip()
    df[context_col] = df[context_col].str.strip()
    df = df[(df[label_col] != "") & (df[context_col] != "")]
    df = df.groupby([label_col, context_col], as_index=False)["n_posts"].sum()

    N = df["n_posts"].sum()
    if N < PMI_MIN_N:
        return pd.DataFrame()

    freq_label = df.groupby(label_col)["n_posts"].sum()
    valid = freq_label[freq_label >= PMI_MIN_N].index
    df = df[df[label_col].isin(valid)]
    if df.empty:
        return pd.DataFrame()

    p_ent = df.groupby(label_col)["n_posts"].sum() / N
    p_ctx = df.groupby(context_col)["n_posts"].sum() / N

    rows = []
    for _, row in df.iterrows():
        ent, ctx, cnt = row[label_col], row[context_col], row["n_posts"]
        p_j = cnt / N
        p_e = p_ent.get(ent, 0)
        p_c = p_ctx.get(ctx, 0)
        if p_e > 0 and p_c > 0 and p_j > 0:
            rows.append({
                label_col:   ent,
                context_col: ctx,
                "pmi":       round(float(np.log2(p_j / (p_e * p_c))), 3),
                "count":     int(cnt),
            })
    return pd.DataFrame(rows)