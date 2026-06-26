"""
FUTURE-Obs — Filtres et calculs sur les données
=================================================
Fonctions stateless (pas de st.*) opérant sur des DataFrames :
  - filter_ctx()   : filtre les ctx_*.csv par façade, saison et objet
  - top_n()        : fréquence descendante, tête de liste
  - compute_pmi()  : matrice PMI entité × contexte
"""

import numpy as np
import pandas as pd

from config import PMI_MIN_N, TOP_N


# ── Filtrage du contexte global ───────────────────────────────────────────────

def filter_ctx(
    ctx: dict,
    facades: list[str],
    saisons: list[str],
    objets: list[str] | None = None,
) -> dict:
    """
    Filtre les DataFrames du dict ctx (produit par load_global_ctx) selon :
      - facades  : liste de façades maritimes (vide = toutes)
      - saisons  : liste de saisons (vide = toutes)
      - objets   : liste d'objets détectés (vide/None = tous)
                   → filtre via ctx["objets"] sur Id_anonym

    Les clés "stats" et les DataFrames vides sont passés tels quels.
    Retourne un nouveau dict (pas de mutation du dict source).
    """
    # Pré-calcul des Id_anonym à conserver si filtre objets actif
    kept_ids: set[str] | None = None
    if objets:
        obj_df = ctx.get("objets", pd.DataFrame())
        if not obj_df.empty and "objet" in obj_df.columns:
            mask = obj_df["objet"].isin([o.lower() for o in objets])
            kept_ids = set(obj_df.loc[mask, "Id_anonym"].astype(str))

    out: dict = {}
    for key, df in ctx.items():
        # "stats" est un dict JSON, pas un DataFrame
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


# ── Top-N fréquences ──────────────────────────────────────────────────────────

def top_n(
    df: pd.DataFrame,
    label_col: str,
    n: int = TOP_N,
) -> pd.DataFrame:
    """
    Retourne les n valeurs les plus fréquentes de label_col.
    Colonnes du résultat : ["label", "count"].
    Retourne un DataFrame vide si df est vide ou si label_col est absent.
    """
    if df.empty or label_col not in df.columns:
        return pd.DataFrame(columns=["label", "count"])

    counts = (
        df[label_col]
        .dropna()
        .str.lower()
        .str.strip()
        .value_counts()
        .head(n)
        .reset_index()
    )
    counts.columns = ["label", "count"]
    return counts


# ── PMI (Pointwise Mutual Information) ───────────────────────────────────────

def compute_pmi(
    df: pd.DataFrame,
    label_col: str,
    context_col: str,
) -> pd.DataFrame:
    """
    Calcule la PMI (log2) entre les entités de label_col et les modalités
    de context_col (typiquement "saison" ou "meta_facade").

    Seules les entités apparaissant au moins PMI_MIN_N fois sont conservées.

    Colonnes du résultat : [label_col, context_col, "pmi", "count"].
    Retourne un DataFrame vide si les données sont insuffisantes.
    """
    df = df[[label_col, context_col]].dropna().copy()
    df[label_col]   = df[label_col].str.lower().str.strip()
    df[context_col] = df[context_col].str.strip()
    df = df[(df[label_col] != "") & (df[context_col] != "")]

    if len(df) < PMI_MIN_N:
        return pd.DataFrame()

    N = len(df)

    # Filtrage des entités trop rares
    freq  = df[label_col].value_counts()
    valid = freq[freq >= PMI_MIN_N].index
    df    = df[df[label_col].isin(valid)]
    if df.empty:
        return pd.DataFrame()

    # Probabilités marginales et jointes
    p_ent  = df[label_col].value_counts()   / N
    p_ctx  = df[context_col].value_counts() / N
    joint  = df.groupby([label_col, context_col]).size() / N

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