"""
FUTURE-Obs — Composants de mise en page
=========================================
Fonctions de rendu du squelette visuel de l'app :
  - header()               : bandeau bleu avec titre et stats
  - section()              : titre de section avec barre bleue
  - show_umap()            : iframe UMAP + bouton agrandir
  - page_umap_fullscreen() : vue plein écran d'un dashboard UMAP
"""

import streamlit as st


# ── Bandeau d'en-tête ─────────────────────────────────────────────────────────

def header(title: str, subtitle: str, stats: dict) -> None:
    """
    Affiche le bandeau bleu foncé en haut de chaque page.

    Paramètres
    ----------
    title    : titre principal (Syne, bold)
    subtitle : ligne de contexte au-dessus du titre (IBM Plex Mono, uppercase)
    stats    : dict ordonné {label: valeur_entière} affiché en ligne sous le titre
    """
    items = "".join(
        f'<div class="stat-item">'
        f'<span class="stat-number">{v:,}</span>'
        f'<span class="stat-label">{k}</span>'
        f'</div>'
        for k, v in stats.items()
    )
    st.markdown(
        f"""
        <div class="site-header">
            <div class="subtitle">{subtitle}</div>
            <h1>{title}</h1>
            <div class="stats-row">{items}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Titre de section ──────────────────────────────────────────────────────────

def section(title: str) -> None:
    """
    Affiche un titre de section avec une barre bleue à gauche.
    À utiliser en début de chaque bloc thématique d'une page.
    """
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


# ── Iframe UMAP ───────────────────────────────────────────────────────────────

def show_umap(html: str, key: str, caption: str = "") -> None:
    """
    Affiche un dashboard UMAP (HTML) en iframe Streamlit.
    Un bouton « Agrandir » déclenche la vue plein écran via session_state.

    Paramètres
    ----------
    html    : contenu HTML complet du dashboard
    key     : identifiant unique pour le bouton (doit être stable entre reruns)
    caption : légende affichée sous l'iframe (optionnel)
    """
    c1, _ = st.columns([1, 8])
    with c1:
        if st.button("⛶ Agrandir", key=f"btn_{key}"):
            st.session_state["umap_fs_html"]    = html
            st.session_state["umap_fs_caption"] = caption
            st.rerun()

    st.iframe(html, height=650)

    if caption:
        st.caption(caption)


# ── Vue plein écran ───────────────────────────────────────────────────────────

def page_umap_fullscreen() -> None:
    """
    Affiche le dashboard UMAP stocké dans st.session_state["umap_fs_html"]
    en plein écran (hauteur 900 px).
    Le bouton « Retour » efface la clé session et relance le routing normal.
    Doit être appelé en priorité dans main() avant tout autre rendu.
    """
    if st.button("← Retour"):
        del st.session_state["umap_fs_html"]
        st.rerun()

    caption = st.session_state.get("umap_fs_caption", "")
    if caption:
        st.caption(caption)

    st.iframe(
        st.session_state["umap_fs_html"],
        height=900,
    )