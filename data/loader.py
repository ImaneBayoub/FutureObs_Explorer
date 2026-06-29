"""
FUTURE-Obs — Chargement des données
=====================================
  - resolve_token()  : résolution du token HuggingFace
  - _hf_csv()        : téléchargement d'un CSV depuis HF
  - load_umap_html() : chargement d'un dashboard UMAP (local puis HF)
  - load_zone()      : chargement d'une zone parc (fichiers bruts allégés)
  - load_global()    : chargement de la zone global (fichiers agrégés + PMI)
  - load_stats()     : lecture du stats.json local (Git)
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from config import HF_REPO, PRECOMPUTED_DIR

# Fichiers agrégés du global
GLOBAL_FILES = {
    # Top 50 par (objet × saison/facade) — classements
    "act_saison":       "ctx_act_top_x_saison.csv",
    "act_facade":       "ctx_act_top_x_facade.csv",
    "imp_saison":       "ctx_imp_top_x_saison.csv",
    "imp_facade":       "ctx_imp_top_x_facade.csv",
    "actor_saison":     "ctx_actor_top_x_saison.csv",
    "actor_facade":     "ctx_actor_top_x_facade.csv",
    # Objets
    "obj_saison":       "ctx_objet_x_saison.csv",
    "obj_facade":       "ctx_objet_x_facade.csv",
    # Carte
    "carte_saison":     "ctx_carte_x_objet_x_saison.csv",
    "carte_facade":     "ctx_carte_x_objet_x_facade.csv",
    # Matrices PMI précalculées (déjà pivotées : label × saison/facade)
    "pmi_act_saison":   "pmi_act_x_saison.csv",
    "pmi_act_facade":   "pmi_act_x_facade.csv",
    "pmi_imp_saison":   "pmi_imp_x_saison.csv",
    "pmi_imp_facade":   "pmi_imp_x_facade.csv",
    "pmi_actor_saison": "pmi_actor_x_saison.csv",
    "pmi_actor_facade": "pmi_actor_x_facade.csv",
    "pmi_obj_saison":   "pmi_objet_x_saison.csv",
    "pmi_obj_facade":   "pmi_objet_x_facade.csv",
}

# Fichiers bruts des parcs
PARC_FILES = {
    "activites":          "activites.csv",
    "impacts_neg":        "impacts_neg.csv",
    "impacts_pos":        "impacts_pos.csv",
    "impacts_neutre":     "impacts_neutre.csv",
    "acteurs_humain":     "acteurs_humain.csv",
    "acteurs_non_humain": "acteurs_non_humain.csv",
    "localisations":      "localisations.csv",
    "ctx_objets":         "ctx_objets.csv",
}


# ── Token HuggingFace ─────────────────────────────────────────────────────────

def resolve_token() -> str | None:
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    _cfg = Path("/home/imane/Documents/FutureObs_Explorer/config/API_keys.json")
    if _cfg.exists():
        try:
            return json.loads(_cfg.read_text())["HF_TOKEN"]
        except Exception:
            pass
    return os.environ.get("HF_TOKEN")


# ── Accès HuggingFace ─────────────────────────────────────────────────────────

def _hf_csv(path_in_repo: str, token: str | None) -> pd.DataFrame:
    local = hf_hub_download(
        repo_id=HF_REPO,
        filename=path_in_repo,
        repo_type="dataset",
        token=token,
    )
    df = pd.read_csv(local, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_umap_html(filename: str, token: str | None) -> str:
    local = Path(__file__).parent.parent / "umap_dashboard" / filename
    if local.exists():
        return local.read_text(encoding="utf-8")
    hf_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"umap_dashboard/{filename}",
        repo_type="dataset",
        token=token,
    )
    return Path(hf_path).read_text(encoding="utf-8")


# ── Global (agrégé + PMI) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=1)
def load_global(token: str | None) -> dict[str, pd.DataFrame]:
    """
    Charge tous les fichiers du global depuis HF.
    Les clés pmi_* contiennent des matrices déjà pivotées
    (label en index, saisons/facades en colonnes, valeurs PMI).
    """
    def safe(filename: str) -> pd.DataFrame:
        try:
            return _hf_csv(f"data_precomputed/global/{filename}", token)
        except Exception:
            return pd.DataFrame()

    return {key: safe(fname) for key, fname in GLOBAL_FILES.items()}


# ── Parcs (bruts allégés) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=2)
def load_zone(zone_slug: str, token: str | None) -> dict[str, pd.DataFrame]:
    """Charge les fichiers bruts allégés d'un parc depuis HF."""
    def safe(filename: str) -> pd.DataFrame:
        try:
            return _hf_csv(f"data_precomputed/{zone_slug}/{filename}", token)
        except Exception:
            return pd.DataFrame()

    return {key: safe(fname) for key, fname in PARC_FILES.items()}


# ── Stats locales (Git) ───────────────────────────────────────────────────────

def load_stats(zone_slug: str) -> dict:
    """Lit <zone_slug>.json depuis stat/ (Git) pour le header Streamlit."""
    path = Path(__file__).parent.parent / "stat" / f"{zone_slug}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}