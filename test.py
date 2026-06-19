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

@st.cache_data(show_spinner="Chargement acteurs…")
def load_acteurs(token):
    path = hf_hub_download(repo_id=HF_REPO, filename="data_clean/acteurs_anonymises.csv", repo_type="dataset", token=token)
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner="Chargement activités…")
def load_activites(token):
    path = hf_hub_download(repo_id=HF_REPO, filename="data_clean/activites_anonymises.csv", repo_type="dataset", token=token)
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner="Chargement impacts…")
def load_impacts(token):
    path = hf_hub_download(repo_id=HF_REPO, filename="data_clean/impacts_anonymises.csv", repo_type="dataset", token=token)
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner="Chargement localisations…")
def load_localisations(token):
    path = hf_hub_download(repo_id=HF_REPO, filename="data_clean/localisations.csv", repo_type="dataset", token=token)
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner="Chargement posts…")
def load_posts(token):
    path = hf_hub_download(repo_id=HF_REPO, filename="data_clean/posts.csv", repo_type="dataset", token=token)
    return pd.read_csv(path, low_memory=False)

hf_token = resolve_token()
if hf_token is None:
    st.warning("⚠️ Aucun token HF trouvé — le dataset doit être public.")

st.write("🟡 chargement acteurs…")
acteurs = load_acteurs(hf_token)
st.write(f"🟢 acteurs : {acteurs.shape}")

st.write("🟡 chargement activités…")
activites = load_activites(hf_token)
st.write(f"🟢 activités : {activites.shape}")

st.write("🟡 chargement impacts…")
impacts = load_impacts(hf_token)
st.write(f"🟢 impacts : {impacts.shape}")

st.write("🟡 chargement localisations…")
localisations = load_localisations(hf_token)
st.write(f"🟢 localisations : {localisations.shape}")

st.write("🟡 chargement posts…")
posts = load_posts(hf_token)
st.write(f"🟢 posts : {posts.shape}")

st.title("FUTURE-Obs")
st.dataframe(posts.head(10))