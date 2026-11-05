# =====================================================================
# APP STREAMLIT - CLASSIFICADOR DE TUMORES DE MAMA
# =====================================================================
# Consome o checkpoint salvo em ../modelos/mama.pt (pesos + arquitetura
# + scaler + métricas de validação).

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer


# ---------------------------------------------------------------------
# 1. ARQUITETURA (idêntica à do treino)
# ---------------------------------------------------------------------
class RedeNeuralRegularizada(nn.Module):
    """MLP binário com Dropout — mesma classe usada no treinamento."""

    def __init__(self, n_entrada, n_oculta1=64, n_oculta2=32, dropout=0.3):
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Linear(n_entrada, n_oculta1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_oculta1, n_oculta2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_oculta2, 1),
        )

    def forward(self, x):
        return self.camadas(x)


# ---------------------------------------------------------------------
# 2. CARREGAMENTO DO MODELO (cacheado)
# ---------------------------------------------------------------------
CAMINHO_MODELO = Path(__file__).parent.parent / 'modelos' / 'mama.pt'


@st.cache_resource
def carregar_modelo(caminho=str(CAMINHO_MODELO)):
    """Carrega o checkpoint e reconstrói modelo + scaler."""
    ckpt = torch.load(caminho, map_location='cpu', weights_only=False)

    arq = ckpt['arquitetura']
    modelo = RedeNeuralRegularizada(
        n_entrada=arq['n_entrada'],
        n_oculta1=arq['n_oculta1'],
        n_oculta2=arq['n_oculta2'],
        dropout=arq['dropout'],
    )
    modelo.load_state_dict(ckpt['state_dict'])
    modelo.eval()

    # Reconstrói o scaler com os parâmetros do treino
    scaler = {
        'mean': np.asarray(ckpt['scaler']['mean']),
        'scale': np.asarray(ckpt['scaler']['scale']),
    }

    return modelo, scaler, ckpt


@st.cache_data
def carregar_features():
    """Nomes das 30 features do dataset."""
    return list(load_breast_cancer().feature_names)


# ---------------------------------------------------------------------
# 3. INFERÊNCIA
# ---------------------------------------------------------------------
def prever(dados_crus, modelo, scaler, limiar=0.5):
    """Recebe dados CRUS (N, n_features) e devolve probabilidades e classes."""
    dados_pad = (np.asarray(dados_crus, dtype=np.float32) - scaler['mean']) / scaler['scale']
    tensor = torch.tensor(dados_pad, dtype=torch.float32)

    modelo.eval()
    with torch.no_grad():
        logits = modelo(tensor)
        probs = torch.sigmoid(logits).numpy().flatten()

    classes = (probs >= limiar).astype(int)
    return probs, classes


# ---------------------------------------------------------------------
# 4. INTERFACE
# ---------------------------------------------------------------------
st.set_page_config(page_title='Classificador de Tumores de Mama',
                   page_icon='🩺', layout='wide')

st.title('🩺 Classificador de Tumores de Mama')
st.caption('Rede neural MLP treinada com o dataset Breast Cancer Wisconsin')

# --- Carrega modelo ---
try:
    modelo, scaler, ckpt = carregar_modelo()
    features = carregar_features()
except FileNotFoundError:
    st.error(f'Modelo não encontrado em `{CAMINHO_MODELO}`. '
             'Rode o notebook de treino antes de iniciar o app.')
    st.stop()

# --- Barra lateral: informações do modelo ---
with st.sidebar:
    st.header('Informações do modelo')

    met = ckpt.get('metricas_validacao', {})
    st.metric('Acurácia (validação)', f"{met.get('acuracia', 0):.1%}")
    st.metric('AUC-ROC (validação)', f"{met.get('auc', 0):.3f}")

    st.divider()
    st.caption(f"**Amostras de treino final:** "
               f"{ckpt.get('treinamento_final', {}).get('n_amostras', '—')}")
    st.caption(f"**Épocas:** {ckpt.get('treinamento_final', {}).get('epocas', '—')}")
    st.caption(f"**Treinado em:** {ckpt.get('treinado_em', '—')}")

    st.divider()
    limiar = st.slider('Limiar de decisão', 0.05, 0.95, 0.50, 0.05,
                       help='Reduza para aumentar a sensibilidade a malignos '
                            '(mais falsos positivos, menos falsos negativos).')

    st.divider()
    st.warning('⚠️ Modelo didático. Não use para diagnóstico real.')

# --- Abas ---
aba_manual, aba_lote, aba_sobre = st.tabs(
    ['Entrada manual', 'Entrada em lote (CSV)', 'Sobre o modelo'])

# =====================================================================
# ABA 1 - ENTRADA MANUAL
# =====================================================================
with aba_manual:
    st.subheader('Informe as 30 medidas do exame')

    with st.form('form_manual'):
        valores = {}

        # Agrupa as features em colunas para não virar uma lista infinita
        colunas = st.columns(3)
        for i, nome in enumerate(features):
            with colunas[i % 3]:
                valores[nome] = st.number_input(
                    nome, value=0.0, format='%.4f', key=f'in_{nome}')

        enviado = st.form_submit_button('Classificar', type='primary')

    if enviado:
        amostra = np.array([[valores[f] for f in features]], dtype=np.float32)
        probs, classes = prever(amostra, modelo, scaler, limiar)
        prob = float(probs[0])
        classe = int(classes[0])

        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            if classe == 1:
                st.success('### Benigno')
            else:
                st.error('### Maligno')

        with col2:
            st.metric('Probabilidade de benigno', f'{prob:.1%}')
            st.progress(prob)

        if prob < limiar:
            st.warning('Resultado abaixo do limiar — classificado como **maligno**. '
                       'Em contexto clínico, exigiria revisão médica.')

# =====================================================================
# ABA 2 - ENTRADA EM LOTE
# =====================================================================
with aba_lote:
    st.subheader('Envie um CSV com as 30 features')

    st.markdown(
        'O arquivo deve ter uma coluna por feature, com os nomes exatos '
        'do dataset. Baixe o modelo de exemplo abaixo se precisar.'
    )

    # Modelo de CSV para download
    exemplo = pd.DataFrame([{f: 0.0 for f in features}])
    st.download_button(
        'Baixar CSV de exemplo',
        exemplo.to_csv(index=False).encode('utf-8'),
        'exemplo_entrada.csv',
        'text/csv',
    )

    arquivo = st.file_uploader('Escolha o arquivo CSV', type=['csv'])

    if arquivo is not None:
        try:
            df_entrada = pd.read_csv(arquivo)

            faltando = [f for f in features if f not in df_entrada.columns]
            if faltando:
                st.error(f'Colunas ausentes no CSV: {faltando[:5]}'
                         f'{"..." if len(faltando) > 5 else ""}')
            else:
                dados = df_entrada[features].values.astype(np.float32)
                probs, classes = prever(dados, modelo, scaler, limiar)

                resultado = df_entrada.copy()
                resultado['prob_benigno'] = probs
                resultado['previsao'] = np.where(classes == 1, 'benigno', 'maligno')

                st.success(f'{len(resultado)} amostras classificadas.')

                col1, col2, col3 = st.columns(3)
                col1.metric('Total', len(resultado))
                col2.metric('Malignos', int((classes == 0).sum()))
                col3.metric('Benignos', int((classes == 1).sum()))

                st.dataframe(resultado, use_container_width=True)

                st.download_button(
                    'Baixar resultados (CSV)',
                    resultado.to_csv(index=False).encode('utf-8'),
                    'resultados.csv',
                    'text/csv',
                )
        except Exception as e:
            st.error(f'Erro ao processar o arquivo: {e}')

# =====================================================================
# ABA 3 - SOBRE O MODELO
# =====================================================================
with aba_sobre:
    st.subheader('Detalhes técnicos')

    st.markdown(f"""
**Arquitetura**

- Entrada: {ckpt['arquitetura']['n_entrada']} features
- Camadas ocultas: {ckpt['arquitetura']['n_oculta1']} → {ckpt['arquitetura']['n_oculta2']}
- Dropout: {ckpt['arquitetura']['dropout']}
- Saída: 1 logit (sigmoid aplicada na inferência)

**Treinamento**

- Retreinado com {ckpt.get('treinamento_final', {}).get('n_amostras', '—')} amostras
  (100% dos dados, após validação)
- Épocas: {ckpt.get('treinamento_final', {}).get('epocas', '—')}
- Métrica de validação: acurácia {met.get('acuracia', 0):.1%},
  AUC {met.get('auc', 0):.3f}

**Pré-processamento**

- Padronização (média 0, desvio 1) com os parâmetros do treino
- O scaler está embutido no checkpoint — não é preciso reajustar

**Limitações**

- Dataset de benchmark (Breast Cancer Wisconsin), não validado clinicamente
- Não substitui avaliação médica
- Sem monitoramento de drift: dados muito diferentes do treino podem
  degradar as previsões
""")