from huggingface_hub import hf_hub_download
import pandas as pd
import streamlit as st
import os
import json
from pathlib import Path

HF_REPO = "ImaneBayoub/future_obs_test"

# Priorité 1 : st.secrets (local .streamlit/secrets.toml + Streamlit Cloud)
# Priorité 2 : JSON local (dev sur machine)
# Priorité 3 : variable d'environnement
HF_TOKEN = st.secrets["HF_TOKEN"]


@st.cache_data(show_spinner="Chargement des données…")
def load_data_hf():
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
            token=HF_TOKEN,
        )
        data[key] = pd.read_csv(local_path, low_memory=False)
    return data

data = load_data_hf()