# ============================================================
# FUNÇÕES AUXILIARES DE SÉRIES TEMPORAIS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from prophet import Prophet
from pmdarima import auto_arima
import xgboost as xgb

import joblib

def formatar_periodo(df, freq=None):
    """
    Retorna o primeiro e último dia do índice formatados conforme a frequência da série.
    
    - Anual → apenas o ano (2009)
    - Mensal → mes/ano (01/2009)
    - Diária → dd/mm/aaaa (01/01/2009)
    - Sub-diária → timestamp completo (01/01/2009 14:30:00)
    """
    if freq is None:
        freq = pd.infer_freq(df.index)
    data_min = df.index.min()
    data_max = df.index.max()
    
    if freq is None:
        # Tenta inferir pela diferença entre observações
        diff = df.index.to_series().diff().median()
        if diff >= pd.Timedelta(days=365):
            freq = 'A'
        elif diff >= pd.Timedelta(days=28):
            freq = 'M'
        elif diff >= pd.Timedelta(days=1):
            freq = 'D'
        else:
            freq = 'H'
    
    # Mapear frequência para formato
    if freq.startswith(('A', 'Y', 'AS', 'YS', 'YE', 'AE')):  # Anual
        fmt = '%Y'
        rotulo = 'anual'
    elif freq.startswith(('M', 'MS', 'ME')):  # Mensal
        fmt = '%m/%Y'
        rotulo = 'mensal'
    elif freq.startswith(('D',)):  # Diária
        fmt = '%d/%m/%Y'
        rotulo = 'diária'
    elif freq.startswith(('B','b',)):  # Diária
        fmt = '%d/%m/%Y'
        rotulo = 'diária sem final de semana'   
    elif freq.startswith(('U','u',)):  # Diária
        fmt = '%d/%m/%Y'
        rotulo = 'dias úteis'         
    elif freq.startswith(('Q', 'QS')):  # Trimestral
        fmt = '%m/%Y'
        rotulo = 'trimestral'
    elif freq.startswith(('W',)):  # Semanal
        fmt = '%d/%m/%Y'
        rotulo = 'semanal'
    else:  # Sub-diária (horária, minutal, etc.)
        fmt = '%d/%m/%Y %H:%M:%S'
        rotulo = 'sub-diária'
    
    inicio = data_min.strftime(fmt)
    fim = data_max.strftime(fmt)
    
    return {
        'frequencia': freq,
        'rotulo': rotulo,
        'inicio': inicio,
        'fim': fim,
        'formato': fmt
    }
  
def verificar_serie(df, frequencia=None, feriados=None,
                    tipo_feriados='nacionais', feriados_modo='append'):
    """
    Verifica a série temporal: tipo do índice, frequência, lacunas,
    duplicatas, múltiplas por período, nulos, negativos e zeros.
    Retorna quantidades E detalhes de cada problema.

    NOVO nesta versão:
    1) feriados_modo: 'append' (feriados informados + biblioteca) ou
       'replace' (apenas os feriados informados).
    2) Feriados só se aplicam a frequências DIÁRIAS ou SUBDIÁRIAS.
    3) Frequências: D, B, U, H, MM, S e combinadas HB, HU, MMB, MMU, SB, SU.
    4) MARCAÇÕES INDEVIDAS: observações fora do conjunto de dias válidos
       (ex.: marcação em sábado quando a frequência é B ou HB).
    """
    # ===== 1. Verifica se é uma Série Temporal =====
    tipo = type(df.index).__name__
    if tipo != 'DatetimeIndex':
        return {
            'mensagem':  '❌ Não foi identificado como uma série temporal',
            'tipo': tipo,
            'frequencia': None,
            'esperado': None,
            'observado': None,
            'lacunas': None,
            'lacunas_datas': None,
            'duplicatas': None,
            'duplicatas_datas': None,
            'multiplas': None,
            'multiplas_detalhe': None,
            'nulos': None,
            'negativos': None,
            'zeros': None,
            'feriados': None,
            'feriados_datas': None,
            'lacunas_reais': None,
            'lacunas_reais_datas': None,
            'feriados_modo': None,
            'feriados_aplicados': None,
            'indevidas': None,
            'indevidas_datas': None
        }

    # ===== 2. Determina a Frequência =====
    if frequencia is None:
        frequencia = pd.infer_freq(df.index)
        if frequencia is None:
            frequencia = 'D'

    # ===== 3. Modo de dias válidos embutido no código =====
    # 'U' → dias úteis sem feriados | 'B' → dias úteis (seg-sex) | 'D' → todos
    if 'U' in frequencia:
        modo_dia = 'U'
    elif 'B' in frequencia:
        modo_dia = 'B'
    else:
        modo_dia = 'D'

    # Frequências em que feriado faz sentido (diária/subdiária)
    FREQ_COM_FERIADO = {'D', 'B', 'U', 'C', 'H', 'h', 'T', 'min', 'MM', 'S',
                        'HB', 'HU', 'MMB', 'MMU', 'SB', 'SU'}
    feriados_aplicados = frequencia in FREQ_COM_FERIADO

    # ===== 4. Constrói o conjunto de feriados (se aplicável) =====
    feriados_dt = pd.DatetimeIndex([])
    if feriados_aplicados:
        feriados_lib = pd.DatetimeIndex([])
        if feriados_modo == 'append':
            try:
                import holidays
                anos = range(df.index.min().year, df.index.max().year + 1)
                if tipo_feriados == 'b3':
                    feriados_set = holidays.BrazilB3(years=anos)
                else:
                    feriados_set = holidays.Brazil(years=anos)
                feriados_lib = pd.DatetimeIndex([pd.Timestamp(d) for d in feriados_set])
            except ImportError:
                feriados_lib = pd.DatetimeIndex([])

        if feriados is not None:
            feriados_user = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in feriados))
        else:
            feriados_user = pd.DatetimeIndex([])

        if feriados_modo == 'replace':
            feriados_dt = feriados_user
        else:
            feriados_dt = feriados_lib.union(feriados_user)

    # ===== 5. Índice esperado conforme a frequência =====
    inicio, fim = df.index.min(), df.index.max()

    if modo_dia == 'U':
        offset = pd.offsets.CustomBusinessDay(holidays=feriados_dt)
        indice_esperado = pd.date_range(start=inicio, end=fim, freq=offset)
    else:
        indice_esperado = pd.date_range(start=inicio, end=fim, freq=frequencia)

    # ===== 6. Lacunas — quais datas estão faltando =====
    lacunas_datas = indice_esperado.difference(df.index)

    feriados_datas = pd.DatetimeIndex([])
    lacunas_reais_datas = lacunas_datas

    if feriados_aplicados and len(lacunas_datas) > 0:
        if frequencia in ('H', 'h', 'T', 'min', 'MM', 'S',
                          'HB', 'HU', 'MMB', 'MMU', 'SB', 'SU'):
            datas_lacunas = lacunas_datas.normalize()
            mascara_feriado = datas_lacunas.isin(feriados_dt)
            feriados_datas = lacunas_datas[mascara_feriado]
            lacunas_reais_datas = lacunas_datas[~mascara_feriado]
        else:
            feriados_datas = lacunas_datas.intersection(feriados_dt)
            lacunas_reais_datas = lacunas_datas.difference(feriados_dt)

    # ===== 7. MARCAÇÕES INDEVIDAS (fora do conjunto de dias válidos) =====
    marcacoes_indevidas_datas = pd.DatetimeIndex([])

    if modo_dia in ('B', 'U'):
        # Fim de semana é sempre indevido para B/U (e combinadas)
        indevidas_fim_semana = df.index[df.index.dayofweek >= 5]

        if modo_dia == 'U':
            # Para U (e combinadas), feriados também são indevidos
            indevidas_feriado = df.index[df.index.normalize().isin(feriados_dt)]
            marcacoes_indevidas_datas = indevidas_fim_semana.union(indevidas_feriado)
        else:
            marcacoes_indevidas_datas = indevidas_fim_semana

        marcacoes_indevidas_datas = marcacoes_indevidas_datas.sort_values()

    # ===== 8. Duplicatas =====
    mascara_duplicatas = df.index.duplicated(keep=False)
    df_duplicatas = df[mascara_duplicatas]
    duplicatas_datas = df_duplicatas.index.unique().sort_values()

    # ===== 9. Múltiplas por período =====
    mapa_periodo = {
        'A': 'A', 'AS': 'A', 'Y': 'A', 'YS': 'A', 'AE': 'A', 'YE': 'A',
        'Q': 'Q', 'QS': 'Q', 'QE': 'Q',
        'M': 'M', 'MS': 'M', 'ME': 'M',
        'W': 'W',
        'D': 'D', 'B': 'D', 'U': 'D', 'C': 'D',
        'H': 'H', 'h': 'H', 'HB': 'H', 'HU': 'H',
        'T': 'T', 'min': 'T', 'MM': 'T', 'MMB': 'T', 'MMU': 'T',
        'S': 'S', 'SB': 'S', 'SU': 'S',
    }
    codigo_periodo = mapa_periodo.get(frequencia, frequencia[0] if frequencia else 'D')
    contagem = df.index.to_period(codigo_periodo).value_counts().sort_index()
    periodos_multiplos = contagem[contagem > 1]

    multiplas_detalhe = []
    for periodo, qtd in periodos_multiplos.items():
        obs = df.loc[periodo.start_time:periodo.end_time]
        for idx, row in obs.iterrows():
            multiplas_detalhe.append({
                'periodo': str(periodo),
                'data': idx,
                'observacoes_no_periodo': qtd,
                'valores': row.to_dict()
            })

    # ===== 10. Nulos, negativos, zeros =====
    nulos = df.isnull().sum().sum()
    negativos = int((df.select_dtypes(include='number') < 0).sum().sum())
    zeros = int((df.select_dtypes(include='number') == 0).sum().sum())

    # ===== 11. Mensagem final =====
    formata = formatar_periodo(df, frequencia)
    mensagem = f"""
## Resumo da EDA

### Qualidade dos Dados
- ✅ É uma Série Temporal
- ✅ Frequência: `{formata['frequencia']} ({formata['rotulo']})`
- ✅ {len(df)} observações ({formata['inicio']} a {formata['fim']})
"""

    # 11.1 Lacunas com distinção de feriados
    if not feriados_aplicados:
        if len(lacunas_datas) == 0:
            mensagem += f"""
- ✅ {len(lacunas_datas)} lacunas de datas
"""
        else:
            mensagem += f"""
- ❌ Esta série tem {len(lacunas_datas)} lacunas no período {list(lacunas_datas.strftime('%d/%m/%Y %H:%M'))[:10]}{'...' if len(lacunas_datas) > 10 else ''}
"""
    elif len(lacunas_reais_datas) == 0 and len(feriados_datas) == 0:
        mensagem += f"""
- ✅ {len(lacunas_datas)} lacunas de datas
"""
    elif len(lacunas_reais_datas) == 0 and len(feriados_datas) > 0:
        mensagem += f"""
- ✅ Sem lacunas reais — {len(feriados_datas)} feriados identificados (não são lacunas)
- ℹ️ Feriados: {list(feriados_datas.strftime('%d/%m/%Y'))[:10]}{'...' if len(feriados_datas) > 10 else ''}
"""
    else:
        mensagem += f"""
- ❌ Esta série tem {len(lacunas_reais_datas)} lacunas reais no período {list(lacunas_reais_datas.strftime('%d/%m/%Y %H:%M'))[:10]}{'...' if len(lacunas_reais_datas) > 10 else ''}
"""
        if len(feriados_datas) > 0:
            mensagem += f"""
- ℹ️ {len(feriados_datas)} feriados ignorados (não são lacunas): {list(feriados_datas.strftime('%d/%m/%Y'))[:10]}{'...' if len(feriados_datas) > 10 else ''}
"""

    # 11.2 Marcações indevidas
    if len(marcacoes_indevidas_datas) == 0:
        mensagem += f"""
- ✅ {len(marcacoes_indevidas_datas)} marcações indevidas
"""
    else:
        mensagem += f"""
- ⚠️ Esta série tem {len(marcacoes_indevidas_datas)} marcações indevidas (fora do conjunto válido para a frequência {frequencia}): {list(marcacoes_indevidas_datas.strftime('%d/%m/%Y %H:%M'))[:10]}{'...' if len(marcacoes_indevidas_datas) > 10 else ''}
"""

    # 11.3 Demais verificações
    if nulos == 0:
        mensagem += f"""
- ✅ {nulos} valores nulos
"""
    else:
        mensagem += f"""
- ❌ Esta série tem valores nulos
"""
    if int(df.index.duplicated().sum()) == 0:
        mensagem += f"""
- ✅ {int(df.index.duplicated().sum())} períodos duplicados
"""
    else:
        mensagem += f"""
- ⚠️ Esta série tem os seguintes períodos repetidos {duplicatas_datas}
"""
    if len(periodos_multiplos) == 0:
        mensagem += f"""
- ✅ {len(periodos_multiplos)} períodos com marcações múltiplas
"""
    else:
        mensagem += f"""
- ⚠️ Esta série tem os seguintes períodos com marcações múltiplas {multiplas_detalhe}
"""

    if negativos == 0:
        mensagem += f"""
- ✅ {negativos} valores negativos
"""
    else:
        mensagem += f"""
- ⚠️ Esta série tem valores negativos
"""
    if zeros == 0:
        mensagem += f"""
- ✅ {zeros} valores zerados
"""
    else:
        mensagem += f"""
- ⚠️ Esta série tem valores zerados
"""

    return {
        'mensagem': mensagem,
        'tipo': tipo,
        'frequencia': frequencia,
        'esperado': len(indice_esperado),
        'observado': len(df),
        'lacunas': len(lacunas_reais_datas),
        'lacunas_datas': list(lacunas_reais_datas.strftime('%d/%m/%Y %H:%M')),
        'duplicatas': int(df.index.duplicated().sum()),
        'duplicatas_datas': list(duplicatas_datas.strftime('%d/%m/%Y')),
        'multiplas': len(periodos_multiplos),
        'multiplas_detalhe': multiplas_detalhe,
        'nulos': int(nulos),
        'negativos': negativos,
        'zeros': zeros,
        'feriados': len(feriados_datas),
        'feriados_datas': list(feriados_datas.strftime('%d/%m/%Y')),
        'lacunas_reais': len(lacunas_reais_datas),
        'lacunas_reais_datas': list(lacunas_reais_datas.strftime('%d/%m/%Y')),
        'feriados_modo': feriados_modo,
        'feriados_aplicados': feriados_aplicados,
        'indevidas': len(marcacoes_indevidas_datas),
        'indevidas_datas': list(marcacoes_indevidas_datas.strftime('%d/%m/%Y %H:%M'))
    }
  
def calcular_metricas(real, previsto):
    mae = np.mean(np.abs(real - previsto))
    rmse = np.sqrt(np.mean((real - previsto) ** 2))
    mape = np.mean(np.abs((real - previsto) / real)) * 100
    smape = np.mean(2 * np.abs(real - previsto) / (np.abs(real) + np.abs(previsto))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sMAPE': smape}
    
# Funções para este problema
def interpretar_acf_pacf(serie_acf, serie_pacf, conf_int, nome):
    """
    Interpreta ACF e PACF para sugerir ordens p, q.
    """
    # Contar lags significativos (acima do intervalo de confiança)
    lags_acf_sig = [i for i in range(1, len(serie_acf)) if abs(serie_acf[i]) > conf_int]
    lags_pacf_sig = [i for i in range(1, len(serie_pacf)) if abs(serie_pacf[i]) > conf_int]

    # Verificar corte abrupto (primeiros k lags significativos, depois cai para zero)
    def detectar_corte(lags_sig, max_lag=10):
        for k in range(1, min(max_lag + 1, len(lags_sig) + 1)):
            if k not in lags_sig:
                return k - 1
        return min(len(lags_sig), max_lag)

    corte_acf = detectar_corte(lags_acf_sig)
    corte_pacf = detectar_corte(lags_pacf_sig)

    # Verificar decaimento exponencial (tail-off)
    def decai(lags_sig, max_lag=10):
        return len(lags_sig) > max_lag

    acf_tail = decai(lags_acf_sig)
    pacf_tail = decai(lags_pacf_sig)

    # Lags sazonais significativos (12, 24, 36)
    lags_sazonais_acf = [l for l in [12, 24, 36] if l < len(serie_acf) and abs(serie_acf[l]) > conf_int]
    lags_sazonais_pacf = [l for l in [12, 24, 36] if l < len(serie_pacf) and abs(serie_pacf[l]) > conf_int]

    return {
        'nome': nome,
        'lags_acf_sig': lags_acf_sig[:10],
        'lags_pacf_sig': lags_pacf_sig[:10],
        'corte_acf': corte_acf,
        'corte_pacf': corte_pacf,
        'acf_tail_off': acf_tail,
        'pacf_tail_off': pacf_tail,
        'lags_sazonais_acf': lags_sazonais_acf,
        'lags_sazonais_pacf': lags_sazonais_pacf,
        'qtd_acf_sig': len(lags_acf_sig),
        'qtd_pacf_sig': len(lags_pacf_sig),
    }
    
def tabela_lags_significativos(acf_serie, pacf_serie, conf_int_val):
    """Gera tabela de lags significativos para ACF e PACF."""
    linhas = []
    for lag in range(1, 11):  # lags 1 a 10
        acf_val = acf_serie[lag] if lag < len(acf_serie) else 0
        pacf_val = pacf_serie[lag] if lag < len(pacf_serie) else 0
        acf_sig = "✅" if abs(acf_val) > conf_int_val else "—"
        pacf_sig = "✅" if abs(pacf_val) > conf_int_val else "—"
        linhas.append(f"| {lag} | {acf_val:+.3f} {acf_sig} | {pacf_val:+.3f} {pacf_sig} |")
    # Lags sazonais
    for lag in [12, 24, 36]:
        if lag < len(acf_serie):
            acf_val = acf_serie[lag]
            pacf_val = pacf_serie[lag]
            acf_sig = "✅" if abs(acf_val) > conf_int_val else "—"
            pacf_sig = "✅" if abs(pacf_val) > conf_int_val else "—"
            linhas.append(f"| **{lag}** (sazonal) | {acf_val:+.3f} {acf_sig} | {pacf_val:+.3f} {pacf_sig} |")
    return "\n".join(linhas)

'''
Dando os devidos créditos:
Esta função foi Baseada no curso de Modelagem de Séries Temporais e Real-Time Analytics 
com Apache Spark e Databricks oferecido pela DSA - Data Science Academy
'''

# Função para testar a estacionaridade
def testa_estacionaridade(serie, window=12, title='Estatísticas Móveis e Teste Dickey-Fuller'):
    """
    Testa a estacionaridade de uma série temporal usando ADF e KPSS.
    Em caso de CONFLITO (ADF indica estacionária + KPSS indica não estacionária),
    roda automaticamente o KPSS com regression='ct' para verificar se a série
    é estacionária em tendência (trend-stationary).

    Parâmetros:
    - serie: pandas.Series. Série temporal a ser testada.
    - window: int. Janela para cálculo das estatísticas móveis.
    - title: str. Título para os gráficos.

    Retorna dict com p-valores (ADF, KPSS nível, KPSS tendência) e conclusão.
    """
    # ===== Estatísticas móveis + gráfico =====
    rolmean = serie.rolling(window=window).mean()
    rolstd = serie.rolling(window=window).std()

    plt.figure(figsize=(14, 6))
    plt.plot(serie, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Média Móvel')
    plt.plot(rolstd, color='black', label='Desvio Padrão Móvel')
    plt.legend(loc='best')
    plt.title(title)
    plt.show(block=False)

    # ===== TESTE ADF =====
    print('\n' + '=' * 62)
    print('TESTE ADF — Dickey-Fuller Aumentado')
    print('H0: a série tem raiz unitária (NÃO estacionária)')
    print('=' * 62)
    dfteste = adfuller(serie, autolag='AIC')
    dfsaida = pd.Series(dfteste[0:4], index=['Estatística do Teste', 'Valor-p',
                                             'Número de Lags Consideradas',
                                             'Número de Observações Usadas'])
    for key, value in dfteste[4].items():
        dfsaida['Valor Crítico (%s)' % key] = value
    print(dfsaida)
    p_adf = float(dfsaida['Valor-p'])

    # ===== TESTE KPSS (nível) =====
    print('\n' + '=' * 62)
    print('TESTE KPSS — nível (regression="c")')
    print('H0: a série é estacionária em nível')
    print('=' * 62)
    kpss_teste = kpss(serie, regression='c', nlags='auto')
    kpss_saida = pd.Series(kpss_teste[0:3], index=['Estatística do Teste', 'Valor-p',
                                                   'Número de Lags Usadas'])
    for key, value in kpss_teste[3].items():
        kpss_saida['Valor Crítico (%s)' % key] = value
    print(kpss_saida)
    p_kpss = float(kpss_saida['Valor-p'])

    # ===== LÓGICA COMBINADA =====
    adf_est = p_adf < 0.05      # ADF rejeita raiz unitária → estacionária
    kpss_est = p_kpss > 0.05    # KPSS não rejeita estacionariedade → estacionária

    p_kpss_ct = None
    kpss_ct_est = None

    # --- Caso 1: CONFLITO trend-stationary (ADF estacionária + KPSS não) ---
    if adf_est and not kpss_est:
        print('\n' + '=' * 62)
        print('⚠️ CONFLITO DETECTADO — ADF estacionária + KPSS não estacionária')
        print('Rodando KPSS com regression="ct" (estacionariedade em tendência)...')
        print('=' * 62)
        kpss_ct = kpss(serie, regression='ct', nlags='auto')
        kpss_ct_saida = pd.Series(kpss_ct[0:3], index=['Estatística do Teste', 'Valor-p',
                                                       'Número de Lags Usadas'])
        for key, value in kpss_ct[3].items():
            kpss_ct_saida['Valor Crítico (%s)' % key] = value
        print(kpss_ct_saida)
        p_kpss_ct = float(kpss_ct_saida['Valor-p'])
        kpss_ct_est = p_kpss_ct > 0.05

        if kpss_ct_est:
            conclusao = 'ESTACIONÁRIA EM TENDÊNCIA (trend-stationary)'
            detalhe = ('O KPSS com tendência NÃO rejeitou a estacionariedade (p > 0.05).\n'
                       'A série é estacionária em torno de uma tendência determinística.\n'
                       '→ Para modelar: inclua a tendência como regressor (ex.: SARIMAX com\n'
                       '  trend="t" ou "ct") OU diferencie a série e modele os resíduos.')
        else:
            conclusao = 'NÃO ESTACIONÁRIA — nem em nível, nem em tendência'
            detalhe = ('O KPSS com tendência TAMBÉM rejeitou a estacionariedade (p < 0.05).\n'
                       'O conflito não é explicado por tendência determinística.\n'
                       '→ Possíveis causas: quebra estrutural (mudança de regime) ou\n'
                       '  integração fracionária. Considere diferenciar a série e re-testar,\n'
                       '  ou testar quebras estruturais (ex.: teste de Chow).')

    # --- Caso 2: CONFLITO inverso (ADF não estacionária + KPSS estacionária) ---
    elif (not adf_est) and kpss_est:
        conclusao = 'CONFLITO — possivelmente integração fracionária'
        detalhe = ('ADF indica não estacionária (p > 0.05), mas KPSS indica estacionária (p > 0.05).\n'
                   'Esse padrão sugere integração fracionária (0 < d < 1), raiz unitária próxima\n'
                   'ou baixo poder dos testes com amostra curta. Considere diferenciar e re-testar.')

    # --- Caso 3: Concordância ---
    elif adf_est and kpss_est:
        conclusao = 'ESTACIONÁRIA'
        detalhe = ('Ambos os testes concordam: a série é estacionária.\n'
                   'ADF rejeitou a raiz unitária (p < 0.05) e KPSS não rejeitou a estacionariedade (p > 0.05).')
    else:
        conclusao = 'NÃO ESTACIONÁRIA'
        detalhe = ('Ambos os testes concordam: a série NÃO é estacionária.\n'
                   'ADF não rejeitou a raiz unitária (p > 0.05) e KPSS rejeitou a estacionariedade (p < 0.05).')

    # ===== RESUMO FINAL =====
    print('\n' + '=' * 62)
    print('CONCLUSÃO COMBINADA (ADF + KPSS)')
    print('=' * 62)
    print(f'ADF        → p = {p_adf:.4f} → {"estacionária" if adf_est else "não estacionária"}')
    print(f'KPSS nível → p = {p_kpss:.4f} → {"estacionária" if kpss_est else "não estacionária"}')
    if p_kpss_ct is not None:
        print(f'KPSS tend. → p = {p_kpss_ct:.4f} → {"estacionária em tendência" if kpss_ct_est else "não estacionária em tendência"}')
    print(f'\nConclusão: {conclusao}')
    print(detalhe)

    return {
        'adf_p': p_adf,
        'kpss_p': p_kpss,
        'kpss_ct_p': p_kpss_ct,
        'adf_estacionaria': adf_est,
        'kpss_estacionaria': kpss_est,
        'kpss_ct_estacionaria': kpss_ct_est,
        'conclusao': conclusao,
        'detalhe': detalhe
    }
    
def decompor_serie(serie, period=None, modelo='additive', plot=True):
    """
    Decompõe a série temporal em tendência, sazonalidade e ruído (resíduo)
    e retorna uma análise quantitativa de cada componente.

    Parâmetros:
    - serie: pandas.Series. Série temporal com DatetimeIndex.
    - period: int. Período sazonal (ex.: 7 para semanal, 12 para mensal).
              Se None, tenta inferir da frequência do índice.
    - modelo: str. 'additive' (padrão) ou 'multiplicative'.
    - plot: bool. Se True, exibe o gráfico da decomposição.

    Retorna:
    - dict com 'mensagem' (análise textual), componentes decompostos,
      forças de tendência/sazonalidade e análise do ruído.
    """
    # 1. Validação básica
    if not isinstance(serie.index, pd.DatetimeIndex):
        return {'mensagem': '❌ A série precisa ter um DatetimeIndex para decomposição.',
                'erro': True}

    serie = serie.dropna()

    # 2. Determina o período sazonal
    if period is None:
        freq = pd.infer_freq(serie.index)
        mapa = {'D': 7, 'B': 5, 'W': 52, 'M': 12, 'MS': 12,
                'Q': 4, 'QS': 4, 'A': 1, 'Y': 1}
        period = mapa.get(freq, 7)   # padrão semanal se não conseguir inferir

    if len(serie) < 2 * period:
        return {'mensagem': f'❌ Série muito curta para decomposição com período {period}. '
                            f'Necessário pelo menos {2 * period} observações.',
                'erro': True}

    # 3. Decomposição
    decomp = seasonal_decompose(serie, model=modelo, period=period)
    tendencia = decomp.trend
    sazonal = decomp.seasonal
    residuo = decomp.resid

    # 4. Forças de tendência e sazonalidade (Wang, Smith & Hyndman, 2006)
    #    Ft = 1 - Var(ruído)/Var(ruído + tendência)  → 0 a 1
    #    Fs = 1 - Var(ruído)/Var(ruído + sazonal)     → 0 a 1
    resid_var = np.var(residuo.dropna(), ddof=1)
    Ft = max(0.0, 1 - resid_var / np.var((tendencia + residuo).dropna(), ddof=1))
    Fs = max(0.0, 1 - resid_var / np.var((sazonal + residuo).dropna(), ddof=1))

    # 5. Análise do ruído (resíduo)
    resid_limpo = residuo.dropna()
    media_resid = resid_limpo.mean()
    std_resid = resid_limpo.std()

    # Teste de Ljung-Box: H0 = resíduos são ruído branco (sem autocorrelação)
    lb = acorr_ljungbox(resid_limpo, lags=[10], return_df=True)
    p_lb = lb['lb_pvalue'].iloc[0]

    # Outliers no resíduo (|z| > 3)
    if std_resid > 0:
        z_resid = np.abs((resid_limpo - media_resid) / std_resid)
        n_outliers = int((z_resid > 3).sum())
    else:
        n_outliers = 0

    # 6. Classificação das forças
    def classificar(forca):
        if forca >= 0.8:
            return 'FORTE'
        elif forca >= 0.5:
            return 'MODERADA'
        else:
            return 'FRACA'

    # 7. Mensagem final
    mensagem = f"""
    
## Análise da Decomposição
### Componentes
- ✅ Modelo: {modelo} | Período sazonal: {period}
- 📈 Tendência: força {Ft:.2f} ({classificar(Ft)})
- 🔁 Sazonalidade: força {Fs:.2f} ({classificar(Fs)})
- 🌫️ Ruído: média {media_resid:.4f}, desvio padrão {std_resid:.4f}
"""
    if p_lb < 0.05:
        mensagem += (f"- ⚠️ Resíduos com autocorrelação significativa "
                     f"(Ljung-Box p={p_lb:.4f}) — há padrão não capturado\n")
    else:
        mensagem += (f"- ✅ Resíduos sem autocorrelação significativa "
                     f"(Ljung-Box p={p_lb:.4f}) — comportamento de ruído branco\n")
    mensagem += f"- 🔍 {n_outliers} outliers no resíduo (|z| > 3)\n"

    # 8. Gráfico da decomposição
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        serie.plot(ax=axes[0], color='black', title='Original')
        tendencia.plot(ax=axes[1], color='red', title='Tendência')
        sazonal.plot(ax=axes[2], color='green', title='Sazonalidade')
        residuo.plot(ax=axes[3], color='blue', title='Ruído (Resíduo)')
        plt.tight_layout()
        plt.show(block=False)

    return {
        'mensagem': mensagem,
        'erro': False,
        'periodo': period,
        'modelo': modelo,
        'tendencia': tendencia,
        'sazonalidade': sazonal,
        'ruido': residuo,
        'forca_tendencia': Ft,
        'forca_sazonalidade': Fs,
        'media_ruido': media_resid,
        'std_ruido': std_resid,
        'ljung_box_p': p_lb,
        'outliers_ruido': n_outliers
    }
    
def treinar_sarima(treino, passos,p=1,d=1,q=1,P=None,D=None,Q=None, s=None):
    # Usamos uma ordem padrão (1,1,1) para baseline financeiro
    if P is None:
        modelo = SARIMAX(treino, order=(p, d, q), enforce_stationarity=False)
    else:
        modelo = SARIMAX(treino, order=(p, d, q), seasonal_order=(P,D,Q,s), enforce_stationarity=False)
    fit_modelo = modelo.fit(disp=False)
    return fit_modelo.forecast(steps=passos).values

def treinar_autoarima(treino, passos):
    # Busca automática do melhor ARIMA. m=5 para capturar o ciclo semanal da bolsa
    modelo = auto_arima(treino, seasonal=True, m=5, stepwise=True, trace=False)
    return modelo.predict(n_periods=passos).values

def treinar_prophet(treino, passos):
    df_train = treino.reset_index()
    df_train.columns = ['ds', 'y']
    df_train['ds'] = df_train['ds'].dt.tz_localize(None) # Remove fuso para evitar erros
    
    modelo = Prophet(daily_seasonality=False, yearly_seasonality=True)
    modelo.fit(df_train)
    
    future = modelo.make_future_dataframe(periods=passos, freq='B')
    forecast = modelo.predict(future)
    return forecast['yhat'].iloc[-passos:].values

def treinar_holt_winters(treino, passos):
    # Tendência aditiva, sem sazonalidade rígida (comum em ações)
    modelo = ExponentialSmoothing(treino, trend='add', seasonal=None, initialization_method="estimated")
    fit_modelo = modelo.fit()
    return fit_modelo.forecast(steps=passos).values

def treinar_theta(treino, passos):
    # Modelo Theta: excelente para capturar a tendência de longo prazo
    modelo = ThetaModel(treino, period=5) # period=5 para dias úteis
    fit_modelo = modelo.fit()
    return fit_modelo.forecast(steps=passos).values

def treinar_xgboost(treino, passos):
    """
    XGBoost requer variáveis estruturadas. Criamos lags (dias anteriores)
    e prevemos iterativamente um passo à frente (Auto-regressão).
    """
    df = treino.to_frame('y')
    lags = 5
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df.dropna(inplace=True)
    
    X_train = df.drop('y', axis=1)
    y_train = df['y']
    
    modelo = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
    modelo.fit(X_train, y_train)
    
    # Previsão iterativa
    historico_recente = list(treino.iloc[-lags:].values)
    previsoes = []
    
    for _ in range(passos):
        X_pred = np.array(historico_recente[-lags:][::-1]).reshape(1, -1)
        pred = modelo.predict(X_pred)[0]
        previsoes.append(pred)
        historico_recente.append(pred)
        
    return np.array(previsoes)
    
def gerar_modelo_producao(serie, nome_modelo):
    """Retreina o modelo vencedor com toda a série e salva."""
    if nome_modelo == 'Auto-ARIMA':
        mod = auto_arima(serie, seasonal=True, m=5, stepwise=True, trace=False)
        previsao_5d = mod.predict(n_periods=5)
    elif nome_modelo == 'Prophet':
        df_full = serie.reset_index()
        df_full.columns = ['ds', 'y']
        df_full['ds'] = df_full['ds'].dt.tz_localize(None)
        mod = Prophet(daily_seasonality=False)
        mod.fit(df_full)
        fut = mod.make_future_dataframe(periods=5, freq='B')
        previsao_5d = mod.predict(fut)['yhat'].iloc[-5:].values
    elif nome_modelo == 'Holt-Winters':
        mod = ExponentialSmoothing(serie, trend='add', seasonal=None, initialization_method="estimated").fit()
        previsao_5d = mod.forecast(5)
    elif nome_modelo == 'Theta':
        mod = ThetaModel(serie, period=5).fit()
        previsao_5d = mod.forecast(5)
    elif nome_modelo == 'SARIMA':
        mod = SARIMAX(serie, order=(1, 1, 1)).fit(disp=False)
        previsao_5d = mod.forecast(5)
    else: # XGBoost
        # Simplificação para produção (pode ser empacotado em uma classe própria)
        df = serie.to_frame('y')
        for i in range(1, 6): df[f'lag_{i}'] = df['y'].shift(i)
        df.dropna(inplace=True)
        mod = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
        mod.fit(df.drop('y', axis=1), df['y'])
        
        hist = list(serie.iloc[-5:].values)
        previsao_5d = []
        for _ in range(5):
            X_pred = np.array(hist[-5:][::-1]).reshape(1, -1)
            pred = mod.predict(X_pred)[0]
            previsao_5d.append(pred)
            hist.append(pred)
            
    # Salvar o modelo fisicamente (serialização)
    arquivo_modelo = f'modelo_{nome_modelo.lower()}_petr4.pkl'
    joblib.dump(mod, arquivo_modelo)
    
    return arquivo_modelo, previsao_5d
