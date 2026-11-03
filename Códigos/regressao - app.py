# ============================================================
# APP STREAMLIT — Previsão de Preço de Imóveis (California Housing)
# Fechamento da aula: leva o modelo treinado para produção
# ============================================================
# Como rodar:  streamlit run app.py
# (na mesma pasta onde estão os arquivos .joblib salvos)

import streamlit as st
import joblib
import numpy as np

# ------------------------------------------------------------
# 1) CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Previsão de Preço de Imóveis",
    page_icon="🏠",
    layout="centered",
)

# ------------------------------------------------------------
# 2) CARREGA MODELO + SCALER (com cache — carrega só 1 vez)
# ------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo final e o scaler salvos no retreino."""
    modelo = joblib.load('../modelos/modelo_final.joblib')
    scaler = joblib.load('../modelos/scaler_final.joblib')
    return modelo, scaler

modelo, scaler = carregar_modelo()

# ------------------------------------------------------------
# 3) MÉTRICAS DO MODELO (da avaliação no teste — antes do retreino)
#    CatBoost: R² 0.8499 | RMSE 0.4434 | MAE 0.2905 (em $100k)
# ------------------------------------------------------------
RMSE = 0.4434   # erro típico em unidades de $100.000
MAE  = 0.2905   # erro médio absoluto em unidades de $100.000
R2   = 0.8499   # variância explicada

# ------------------------------------------------------------
# 4) INTERFACE — FORMULÁRIO COM AS 8 FEATURES ORIGINAIS
#    (as 3 features derivadas são calculadas automaticamente)
# ------------------------------------------------------------
st.title("🏠 Previsão de Preço de Imóveis")
st.caption(
    f"Modelo final: **CatBoostRegressor** — R² {R2:.4f} | "
    f"treinado com 100% dos dados (20.640 quarteirões)"
)

st.markdown("### 📋 Informe as características do quarteirão")

with st.form("form_imovel"):
    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            "Renda média do quarteirão (US$ 10 mil)",
            min_value=0.5, max_value=15.0, value=3.87, step=0.01,
            help="MedInc — renda média em dezenas de milhares de dólares",
        )
        house_age = st.number_input(
            "Idade média das casas (anos)",
            min_value=1, max_value=52, value=29, step=1,
        )
        ave_rooms = st.number_input(
            "Média de cômodos por domicílio",
            min_value=0.85, max_value=141.91, value=5.43, step=0.01,
        )
        ave_bedrms = st.number_input(
            "Média de quartos por domicílio",
            min_value=0.33, max_value=34.07, value=1.10, step=0.01,
        )

    with col2:
        population = st.number_input(
            "População do quarteirão",
            min_value=3, max_value=35682, value=1425, step=1,
        )
        ave_occup = st.number_input(
            "Média de ocupantes por domicílio",
            min_value=0.69, max_value=1243.33, value=3.07, step=0.01,
        )
        latitude = st.number_input(
            "Latitude",
            min_value=32.54, max_value=41.95, value=34.26, step=0.01,
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-124.35, max_value=-114.31, value=-118.49, step=0.01,
        )

    submitted = st.form_submit_button("🔮 Prever preço", type="primary")

# ------------------------------------------------------------
# 5) CÁLCULO DA PREVISÃO (só quando o botão é clicado)
# ------------------------------------------------------------
if submitted:
    # 5.1 Feature engineering — MESMAS razões criadas no notebook!
    #     (a ordem das colunas precisa ser idêntica à do treino)
    rooms_per_person   = ave_rooms / ave_occup
    income_per_person  = med_inc / ave_occup
    bedrooms_per_room  = ave_bedrms / ave_rooms

    # 5.2 Monta o vetor na MESMA ordem do X do treino
    X_novo = np.array([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude,
        rooms_per_person, income_per_person, bedrooms_per_room,
    ]])

    # 5.3 Escala (mesmo scaler do treino) e prevê
    X_scaled = scaler.transform(X_novo)
    pred = modelo.predict(X_scaled)[0]   # em unidades de $100.000

    # 5.4 Converte para valores reais
    valor_usd = pred * 100_000
    erro_usd  = RMSE * 100_000           # faixa de erro em dólares
    erro_pct  = (RMSE / pred) * 100      # percentual de erro relativo

    # 5.5 Exibe o resultado
    st.divider()
    st.markdown("## 📊 Resultado da Previsão")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Preço previsto", f"US$ {valor_usd:,.0f}")
    col_b.metric("Erro típico (±)", f"± US$ {erro_usd:,.0f}")
    col_c.metric("Erro relativo", f"± {erro_pct:.1f}%")

    st.info(
        f"**Faixa de confiança:** o valor real deve estar entre "
        f"**US$ {valor_usd - erro_usd:,.0f}** e "
        f"**US$ {valor_usd + erro_usd:,.0f}** "
        f"(1 desvio padrão do erro do modelo)."
    )

    st.caption(
        f"⚠️ O percentual de erro é calculado como RMSE ÷ previsão. "
        f"Quanto menor o preço previsto, maior o percentual — "
        f"o erro absoluto (± US$ {erro_usd:,.0f}) é constante."
    )