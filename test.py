import streamlit as st
st.write("🟡 imports de base ok")

import pandas as pd
import os
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
st.write("🟡 tous les imports ok")

HF_REPO = "ImaneBayoub/future_obs_test"

def resolve_token():
    try:
        token = st.secrets["HF_TOKEN"]
        st.write("🟢 token trouvé via st.secrets")
        return token
    except Exception as e:
        st.write(f"🔴 st.secrets échoué : {e}")
    _config = Path("/home/imane/Documents/FutureObs_Explorer/config/API_keys.json")
    if _config.exists():
        try:
            token = json.loads(_config.read_text())["HF_TOKEN"]
            st.write("🟢 token trouvé via JSON local")
            return token
        except Exception as e:
            st.write(f"🔴 JSON local échoué : {e}")
    token = os.environ.get("HF_TOKEN")
    if token:
        st.write("🟢 token trouvé via variable d'environnement")
    else:
        st.write("🔴 aucun token trouvé")
    return token

@st.cache_data(show_spinner="Chargement des données…")
def load_data_hf(token):
    fichiers = {
        "acteurs":       "data_clean/acteurs_anonymises.csv",
        "activites":     "data_clean/activites_anonymises.csv",
        "impacts":       "data_clean/impacts_anonymises.csv",
        "localisations": "data_clean/localisations.csv",
        "posts":         "data_clean/posts.csv",
    }
    data = {}
    for key, path_in_repo in fichiers.items():
        st.write(f"🟡 téléchargement : {key}…")
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            token=token,
        )
        st.write(f"🟢 téléchargé : {key} → {local_path}")
        data[key] = pd.read_csv(local_path, low_memory=False)
        st.write(f"🟢 chargé : {key} {pd.read_csv(local_path, low_memory=False).shape}")
    return data

hf_token = resolve_token()
if hf_token is None:
    st.warning("⚠️ Aucun token HF trouvé — le dataset doit être public.")

data = load_data_hf(hf_token)
st.write("🟢 toutes les données chargées", {k: v.shape for k, v in data.items()})