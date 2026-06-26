"""
FUTURE-Obs — Chargement des données
=====================================
Toutes les fonctions d'accès aux données :
  - resolve_token()  : résolution du token HuggingFace
  - _hf_csv()        : téléchargement d'un CSV depuis HF
  - load_umap_html() : chargement d'un dashboard UMAP (local puis HF)
  - load_zone()      : chargement d'une zone (parc ou global) depuis HF
  - load_stats()     : lecture du stats.json local (Git)

Nouvelle structure HF :
  data_precomputed/<zone>/
    activites.csv, impacts_neg.csv, impacts_pos.csv, impacts_neutre.csv,
    acteurs_humain.csv, acteurs_non_humain.csv, localisations.csv, ctx_objets.csv

stats.json par zone lus en local depuis data_precomputed/<zone>/stats.json (Git).
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from config import HF_REPO, PRECOMPUTED_DIR

# Fichiers attendus par zone sur HF
ZONE_FILES = [
    "activites.csv",
    "impacts_neg.csv",
    "impacts_pos.csv",
    "impacts_neutre.csv",
    "acteurs_humain.csv",
    "acteurs_non_humain.csv",
    "localisations.csv",
    "ctx_objets.csv",
]


# ── Token HuggingFace ─────────────────────────────────────────────────────────

def resolve_token() -> str | None:
    """
    Résout le token HF dans l'ordre de priorité :
      1. st.secrets["HF_TOKEN"]  (Streamlit Cloud ou secrets.toml local)
      2. /home/imane/Documents/FutureObs_Explorer/config/API_keys.json
      3. Variable d'environnement HF_TOKEN
    """
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
    import os
    return os.environ.get("HF_TOKEN")


# ── Accès HuggingFace ─────────────────────────────────────────────────────────

def _hf_csv(path_in_repo: str, token: str | None) -> pd.DataFrame:
    """Télécharge un CSV depuis HF et le retourne en DataFrame."""
    local = hf_hub_download(
        repo_id=HF_REPO,
        filename=path_in_repo,
        repo_type="dataset",
        token=token,
    )
    return pd.read_csv(local, low_memory=False)


@st.cache_data(show_spinner=False)
def load_umap_html(filename: str, token: str | None) -> str:
    """
    Charge un dashboard UMAP.
    Cherche d'abord en local (umap_dashboard/), puis fallback HF.
    """
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


# ── Chargement d'une zone (parc ou global) ────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=2)
def load_zone(zone_slug: str, token: str | None) -> dict[str, pd.DataFrame]:
    """
    Charge tous les fichiers d'une zone depuis HF (data_precomputed/<zone_slug>/).
    Fonctionne de façon identique pour "global" et n'importe quel parc.

    Retourne un dict avec les clés :
      "activites", "impacts_neg", "impacts_pos", "impacts_neutre",
      "acteurs_humain", "acteurs_non_humain", "localisations", "ctx_objets"
    Chaque valeur est un DataFrame (vide si le fichier est absent ou illisible).
    """
    def safe(filename: str) -> pd.DataFrame:
        try:
            df = _hf_csv(f"data_precomputed/{zone_slug}/{filename}", token)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            return pd.DataFrame()

    return {
        "activites":        safe("activites.csv"),
        "impacts_neg":      safe("impacts_neg.csv"),
        "impacts_pos":      safe("impacts_pos.csv"),
        "impacts_neutre":   safe("impacts_neutre.csv"),
        "acteurs_humain":   safe("acteurs_humain.csv"),
        "acteurs_non_humain": safe("acteurs_non_humain.csv"),
        "localisations":    safe("localisations.csv"),
        "ctx_objets":       safe("ctx_objets.csv"),
    }


# ── Stats locales (Git) ───────────────────────────────────────────────────────

def load_stats(zone_slug: str) -> dict:
    """
    Lit le stats.json d'une zone depuis data_precomputed/<zone_slug>/stats.json.
    Ce fichier est versionné dans Git, pas sur HF.
    Retourne un dict vide si absent ou illisible.
    """
    path = PRECOMPUTED_DIR / zone_slug / "stats.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}