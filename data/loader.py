"""
FUTURE-Obs — Chargement des données
=====================================
  - resolve_token()  : résolution du token HuggingFace
  - _hf_csv()        : téléchargement d'un CSV depuis HF
  - load_umap_html() : chargement d'un dashboard UMAP (local puis HF)
  - load_zone()      : chargement d'une zone parc (fichiers bruts allégés)
  - load_global()    : chargement de la zone global (fichiers agrégés)
  - load_stats()     : lecture du stats.json local (Git)

Structure HF :
  data_precomputed/global/
    ctx_act_x_objet_x_saison.csv, ctx_act_x_objet_x_facade.csv
    ctx_imp_x_objet_x_saison.csv, ctx_imp_x_objet_x_facade.csv
    ctx_actor_x_objet_x_saison.csv, ctx_actor_x_objet_x_facade.csv
    ctx_objet_x_saison.csv, ctx_objet_x_facade.csv
    ctx_carte_x_objet_x_saison.csv, ctx_carte_x_objet_x_facade.csv

  data_precomputed/<parc>/
    activites.csv, impacts_neg.csv, impacts_pos.csv, impacts_neutre.csv,
    acteurs_humain.csv, acteurs_non_humain.csv, localisations.csv, ctx_objets.csv
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
    "act_saison":          "ctx_act_x_objet_x_saison.csv",
    "act_facade":          "ctx_act_x_objet_x_facade.csv",
    "imp_saison":          "ctx_imp_x_objet_x_saison.csv",
    "imp_facade":          "ctx_imp_x_objet_x_facade.csv",
    "actor_hum_saison":    "ctx_actor_humain_x_objet_x_saison.csv",
    "actor_hum_facade":    "ctx_actor_humain_x_objet_x_facade.csv",
    "actor_nh_saison":     "ctx_actor_non_humain_x_objet_x_saison.csv",
    "actor_nh_facade":     "ctx_actor_non_humain_x_objet_x_facade.csv",
    "obj_saison":          "ctx_objet_x_saison.csv",
    "obj_facade":          "ctx_objet_x_facade.csv",
    "carte_saison":        "ctx_carte_x_objet_x_saison.csv",
    "carte_facade":        "ctx_carte_x_objet_x_facade.csv",
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


# ── Global (agrégé) ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=1)
def load_global(token: str | None) -> dict[str, pd.DataFrame]:
    """
    Charge les fichiers agrégés du global depuis HF.
    Retourne un dict avec les clés de GLOBAL_FILES.
    Chaque DataFrame a les colonnes (label, objet, saison/facade, n_posts).
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
    """
    Charge les fichiers bruts allégés d'un parc depuis HF.
    Retourne un dict avec les clés de PARC_FILES.
    """
    def safe(filename: str) -> pd.DataFrame:
        try:
            return _hf_csv(f"data_precomputed/{zone_slug}/{filename}", token)
        except Exception:
            return pd.DataFrame()

    return {key: safe(fname) for key, fname in PARC_FILES.items()}


# ── Stats locales (Git) ───────────────────────────────────────────────────────

def load_stats(zone_slug: str) -> dict:
    """Lit stats.json depuis data_precomputed/<zone_slug>/stats.json (Git)."""
    path = PRECOMPUTED_DIR / zone_slug / "stats.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}