"""
FUTURE-Obs — Configuration globale
=====================================
Toutes les constantes partagées entre les modules de l'application.
Aucun import streamlit ici pour garder ce fichier importable hors contexte Streamlit.
"""

from pathlib import Path

# ── Repo HuggingFace ──────────────────────────────────────────────────────────

HF_REPO = "medialab-sciencespo/FutureObs"

# ── Parcs naturels marins ─────────────────────────────────────────────────────

ZONES: dict[str, str] = {
    "Golfe du Lion":                "golfe_du_lion",
    "Iroise":                       "iroise",
    "Méditerranée - sites côtiers": "mediterranee_sites",
    "Pertuis et Gironde":           "pertuis_gironde",
    "Bassin d'Arcachon":            "bassin_arcachon",
    "Estuaires picards":            "estuaires_picards",
}

# ── Filtres ───────────────────────────────────────────────────────────────────

FACADES_VALIDES: list[str] = [
    "Méditerranée",
    "Nord Atlantique - Manche Ouest",
    "Manche Est - mer du Nord",
    "Sud Atlantique",
]

SAISONS: list[str] = ["printemps", "été", "automne", "hiver"]

# ── Paramètres d'analyse ──────────────────────────────────────────────────────

TOP_N:     int = 25   # nombre d'entités affichées par défaut dans les top-N
PMI_MIN_N: int = 3    # fréquence minimale pour qu'une entité entre dans le calcul PMI

# ── Chemins locaux ────────────────────────────────────────────────────────────

# Répertoire des CSV précalculés (à côté de app.py)
PRECOMPUTED_DIR: Path = Path(__file__).parent / "data_precomputed"

# ── Couleurs ──────────────────────────────────────────────────────────────────

PLATFORM_COLORS: dict[str, str] = {
    "FB":        "#1877F2",
    "facebook":  "#1877F2",
    "IG":        "#E1306C",
    "instagram": "#E1306C",
    "YT":        "#FF0000",
    "youtube":   "#FF0000",
    "LKN":       "#0A66C2",
    "linkedin":  "#0A66C2",
    "TKT":       "#69C9D0",
    "tiktok":    "#69C9D0",
    "X":         "#14171A",
    "twitter":   "#1DA1F2",
}

ZONE_COLORS: dict[str, str] = {
    "golfe_du_lion":      "#4169E1",
    "iroise":             "#20B2AA",
    "mediterranee_sites": "#DC143C",
    "pertuis_gironde":    "#3CB371",
    "bassin_arcachon":    "#FF8C00",
    "estuaires_picards":  "#9370DB",
}