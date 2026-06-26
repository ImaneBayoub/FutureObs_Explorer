"""
FUTURE-Obs — Filtres et calculs sur les données
=================================================
  - filter_agg()   : filtre les fichiers agrégés globaux (objet, saison, facade)
  - filter_ctx()   : filtre les fichiers bruts parcs (Id_anonym, saison, facade)
  - top_n()        : fréquence descendante sur fichiers bruts
  - top_n_agg()    : classement depuis fichiers agrégés (somme n_posts)
  - compute_pmi()  : matrice PMI depuis fichiers bruts
  - compute_pmi_agg() : matrice PMI depuis fichiers agrégés
"""

import numpy as np
import pandas as pd

from config import PMI_MIN_N, TOP_N


# ── Filtrage fichiers agrégés (global) ────────────────────────────────────────

def filter_agg(
    ctx: dict[str, pd.DataFrame],
    objets:  list[str] | None = None,
    saisons: list[str] | None = None,
    facades: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Filtre les DataFrames agrégés du global.
    - objets  : filtre sur colonne "objet"  (None ou [] = tous, y compris __all__)
    - saisons : filtre sur colonne "saison" dans les fichiers *_saison
    - facades : filtre sur colonne "meta_facade" dans les fichiers *_facade

    Si objets est vide/None → garde uniquement les lignes objet == "__all__"
    (comptages sans filtre objet, évite de doubler les n_posts).
    Si objets est renseigné → garde uniquement les lignes correspondantes
    (exclut __all__).
    """
    out: dict[str, pd.DataFrame] = {}

    for key, df in ctx.items():
        if df.empty:
            out[key] = df
            continue

        filtered = df.copy()

        # Filtre objet
        if "objet" in filtered.columns:
            if objets:
                filtered = filtered[filtered["objet"].isin([o.lower() for o in objets])]
            else:
                filtered = filtered[filtered["objet"] == "__all__"]

        # Filtre saison (fichiers *_saison)
        if saisons and "saison" in filtered.columns:
            filtered = filtered[filtered["saison"].isin(saisons)]

        # Filtre facade (fichiers *_facade)
        if facades and "meta_facade" in filtered.columns:
            filtered = filtered[filtered["meta_facade"].isin(facades)]

        out[key] = filtered

    return out


# ── Filtrage fichiers bruts (parcs) ───────────────────────────────────────────

def filter_ctx(
    ctx: dict,
    facades: list[str],
    saisons: list[str],
    objets:  list[str] | None = None,
) -> dict:
    """
    Filtre les DataFrames bruts d'un parc.
    Le filtre objet passe par ctx_objets → Id_anonym → autres fichiers.
    """
    kept_ids: set[str] | None = None
    if objets:
        obj_df = ctx.get("objets", ctx.get("ctx_objets", pd.DataFrame()))
        if not obj_df.empty and "objet" in obj_df.columns:
            mask = obj_df["objet"].isin([o.lower() for o in objets])
            kept_ids = set(obj_df.loc[mask, "Id_anonym"].astype(str))

    out: dict = {}
    for key, df in ctx.items():
        if key == "stats" or not isinstance(df, pd.DataFrame) or df.empty:
            out[key] = df
            continue

        filtered = df.copy()

        if facades and "meta_facade" in filtered.columns:
            filtered = filtered[filtered["meta_facade"].isin(facades)]

        if saisons and "saison" in filtered.columns:
            filtered = filtered[filtered["saison"].isin(saisons)]

        if kept_ids is not None and "Id_anonym" in filtered.columns:
            filtered = filtered[filtered["Id_anonym"].astype(str).isin(kept_ids)]

        out[key] = filtered

    return out


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
    """
    Top-N depuis fichiers agrégés (somme de n_posts par label).
    Les fichiers ont déjà été filtrés par filter_agg().
    """
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
    """PMI depuis fichiers bruts (calcul sur les lignes individuelles)."""
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
    """
    PMI depuis fichiers agrégés (n_posts comme poids).
    df doit avoir colonnes : label_col, context_col, n_posts.
    Les fichiers ont déjà été filtrés par filter_agg().
    """
    if df.empty or label_col not in df.columns or "n_posts" not in df.columns:
        return pd.DataFrame()
    if context_col not in df.columns:
        return pd.DataFrame()

    df = df[[label_col, context_col, "n_posts"]].dropna().copy()
    df[label_col]   = df[label_col].str.lower().str.strip()
    df[context_col] = df[context_col].str.strip()
    df = df[(df[label_col] != "") & (df[context_col] != "")]

    # Agréger les n_posts pour ce couple (label, context)
    df = df.groupby([label_col, context_col], as_index=False)["n_posts"].sum()

    N = df["n_posts"].sum()
    if N < PMI_MIN_N:
        return pd.DataFrame()

    # Filtrage entités trop rares
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