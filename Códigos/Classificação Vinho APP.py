"""
App de classificação de vinhos — consome o modelo treinado no notebook.
Rode com:  streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Classificador de Vinhos", page_icon="🍷", layout="wide")

# ─────────────────────────────────────────────────────────────
# Carrega o artefato (pipeline + metadados) — cacheado
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    caminho = Path(__file__).parent / "../modelos" / "wine_qda.joblib"
    return joblib.load(caminho)

art = carregar_modelo()
pipeline = art["pipeline"]
FEATURES = art["features"]        # ordem exata exigida pelo modelo
CLASSES = art["classes"]          # mapeamento das classes (0, 1, 2)

NOMES_CLASSES = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}

# ─────────────────────────────────────────────────────────────
# Faixas dos sliders (mín/máx realistas do dataset Wine)
# ─────────────────────────────────────────────────────────────
FAIXAS = {
    "alcohol":            (11.0, 15.0, 13.0),
    "malic_acid":         (0.7, 5.8, 2.3),
    "ash":                (1.4, 3.2, 2.3),
    "alcalinity_of_ash":  (10.6, 30.0, 19.5),
    "magnesium":          (70.0, 162.0, 99.0),
    "total_phenols":      (0.98, 3.9, 2.3),
    "flavanoids":         (0.34, 5.1, 2.0),
    "nonflavanoid_phenols": (0.13, 0.66, 0.36),
    "proanthocyanins":    (0.41, 3.6, 1.6),
    "color_intensity":    (1.3, 13.0, 5.0),
    "hue":                (0.48, 1.0, 0.96),
    "od280_od315":        (1.27, 4.0, 2.6),
    "proline":            (280.0, 1680.0, 750.0),
}

# ─────────────────────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────────────────────
st.title("🍷 Classificador de Vinhos")
st.caption(f"Modelo: {art['nome']} · F1-macro (teste): {art.get('f1_macro_teste', '—')} · "
           f"treinado em {art.get('criado_em', '—')[:10]}")

col1, col2, col3 = st.columns(3)

valores = {}
for i, feat in enumerate(FEATURES):
    alvo = [col1, col2, col3][i % 3]
    min_v, max_v, padrao = FAIXAS.get(feat, (0.0, 10.0, 1.0))
    valores[feat] = alvo.slider(
        feat, min_value=float(min_v), max_value=float(max_v),
        value=float(padrao), step=(max_v - min_v) / 100
    )

st.divider()

if st.button("🔬 Classificar amostra", type="primary", use_container_width=True):
    amostra = pd.DataFrame([valores])[FEATURES]   # garante a ordem do contrato
    predicao = pipeline.predict(amostra)[0]
    probs = pipeline.predict_proba(amostra)[0]

    st.subheader(f"Predição: **{NOMES_CLASSES.get(predicao, predicao)}**")

    # Probabilidades por classe
    df_probs = pd.DataFrame({
        "Classe": [NOMES_CLASSES.get(c, c) for c in CLASSES],
        "Probabilidade": probs.round(4),
    }).set_index("Classe")
    st.bar_chart(df_probs)

    with st.expander("Dados enviados ao modelo"):
        st.write(amostra.T.rename(columns={0: "valor"}))