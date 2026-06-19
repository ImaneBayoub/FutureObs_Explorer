import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

HF_REPO = "ImaneBayoub/future_obs_test"

def resolve_token():
    # 1. st.secrets (Streamlit Cloud)
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    # 2. JSON local (dev)
    _config = Path("/home/imane/Documents/FutureObs_Explorer/config/API_keys.json")
    if _config.exists():
        try:
            return json.loads(_config.read_text())["HF_TOKEN"]
        except Exception:
            pass
    # 3. Variable d'environnement
    return os.environ.get("HF_TOKEN")

@st.cache_data(show_spinner="Chargement des données…")
def load_data_hf(token):   # ← token passé en argument pour que le cache soit lié au token
    fichiers = {
        "acteurs":       "data_clean/acteurs_anonymises.csv",
        "activites":     "data_clean/activites_anonymises.csv",
        "impacts":       "data_clean/impacts_anonymises.csv",
        "localisations": "data_clean/localisations.csv",
        "posts":         "data_clean/posts.csv",
    }
    data = {}
    for key, path_in_repo in fichiers.items():
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            token=token,
        )
        data[key] = pd.read_csv(local_path, low_memory=False)
    return data

# ← Tout ici s'exécute dans le contexte Streamlit
hf_token = resolve_token()
if hf_token is None:
    st.warning(" Aucun token HF trouvé — le dataset doit être public.")

data = load_data_hf(hf_token)