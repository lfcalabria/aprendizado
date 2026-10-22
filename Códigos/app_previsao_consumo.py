# ============================================================
# APP STREAMLIT — PREVISÃO DE CONSUMO (SARIMAX SALVO)
# Entradas na tela: data, temperatura, volume de chuva
# Saída: consumo previsto (com intervalo de confiança)
# ============================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title='Previsão de Consumo', page_icon='🍺', layout='centered')

# --- Carrega o modelo salvo (em cache: carrega só na 1ª execução) ---
@st.cache_resource
def carregar_modelo(caminho='../modelos/modelo_sarimax_consumo.pkl'):
    return joblib.load(caminho)

try:
    pacote = carregar_modelo()
except FileNotFoundError:
    st.error("Arquivo 'modelo_sarimax_consumo.pkl' não encontrado. "
             "Execute primeiro o script de treinamento e deixe o .pkl nesta pasta.")
    st.stop()

modelo = pacote['modelo']
features_exog = pacote['features_exog']                      # ['temp', 'chuva', 'fds']
ultima_data_treino = pd.Timestamp(pacote['ultima_data_treino'])
data_minima = (ultima_data_treino + pd.Timedelta(days=1)).date()

# --- Tela ---
st.title('🍺 Previsão de Consumo de Cerveja')
st.caption(f'Modelo SARIMAX treinado até {ultima_data_treino.date()} · '
           f'MAE teste: {pacote["metricas"]["mae_teste"]:.2f}')

col1, col2, col3 = st.columns(3)
with col1:
    data = st.date_input('📅 Data da previsão', value=data_minima, min_value=data_minima)
with col2:
    temp = st.number_input('🌡️ Temperatura máx. (°C)', min_value=-10.0, max_value=50.0,
                           value=25.0, step=0.1)
with col3:
    chuva = st.number_input('🌧️ Volume de chuva (mm)', min_value=0.0, max_value=100.0,
                            value=0.0, step=0.1)

# 'fds' NÃO é digitada: deriva da data, com a MESMA regra do treino
# (weekday() >= 5 → Sábado=5, Domingo=6)
fds = 1 if data.weekday() >= 5 else 0
st.caption('📆 Dia de fim de semana' if fds == 1 else '📆 Dia útil')

if st.button('🔮 Prever consumo', type='primary'):
    # Quantos passos à frente da última data de treino?
    n_passos = (data - ultima_data_treino.date()).days
    if n_passos < 1:
        n_passos = 1

    # Monta as exógenas futuras: uma linha por dia entre a última data
    # de treino e a data escolhida. Temperatura e chuva informadas são
    # repetidas nos dias intermediários (aproximação documentada).
    datas_futuras = pd.date_range(data_minima, periods=n_passos, freq='D')
    df_exog_futuro = pd.DataFrame({
        'temp': [temp] * n_passos,
        'chuva': [chuva] * n_passos,
        'fds': [1 if d.weekday() >= 5 else 0 for d in datas_futuras],
    })[features_exog]  # garante a ordem exata do treinamento

    # Previsão com intervalo de confiança
    previsao, intervalo = modelo.predict(n_periods=n_passos,
                                         X=df_exog_futuro,
                                         return_conf_int=True)

    consumo = float(np.asarray(previsao).ravel()[-1])
    lim_inf, lim_sup = np.asarray(intervalo)[-1]

    st.metric(label='Consumo previsto', value=f'{consumo:.2f} litros')
    st.caption(f'Intervalo de confiança (95%): {lim_inf:.2f} a {lim_sup:.2f} litros')