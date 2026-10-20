# ============================================================
# FUNÇÕES AUXILIARES DE SÉRIES TEMPORAIS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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
    
def verificar_serie(df, frequencia=None):
    """
    Verifica a série temporal: tipo do índice, frequência, lacunas,
    duplicatas, múltiplas por período, nulos, negativos e zeros.
    Retorna quantidades E detalhes de cada problema.
    """
    # 1. Verifica se é uma Série Temporal
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
            'zeros': None
        } 
    
    # 2. Determina a Frequência 
    if frequencia is None:
        frequencia = pd.infer_freq(df.index)
        if frequencia is None:
            frequencia = 'D'

    # 3. Total esperado vs observado
    indice_esperado = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=frequencia
    )

    # 4. Lacunas — quais datas estão faltando
    lacunas_datas = indice_esperado.difference(df.index)

    # 5. Duplicatas — quais datas estão repetidas
    mascara_duplicatas = df.index.duplicated(keep=False)  
    df_duplicatas = df[mascara_duplicatas]
    duplicatas_datas = df_duplicatas.index.unique().sort_values()

    # 6. Múltiplas por período — quais períodos têm mais de 1 observação
    mapa_periodo = {
        'A': 'A', 'AS': 'A', 'Y': 'A', 'YS': 'A', 'AE': 'A', 'YE': 'A',
        'Q': 'Q', 'QS': 'Q', 'QE': 'Q',
        'M': 'M', 'MS': 'M', 'ME': 'M',
        'W': 'W',
        'D': 'D',
        'H': 'H', 'h': 'H',
        'T': 'T', 'min': 'T',
        'S': 'S',
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

    # 7. Nulos
    nulos = df.isnull().sum().sum()

    # 8. Negativos
    negativos = int((df.select_dtypes(include='number') < 0).sum().sum())

    # 9. Zeros
    zeros = int((df.select_dtypes(include='number') == 0).sum().sum())

    # 10 Gerando a mensagem final
    formata = formatar_periodo(df, frequencia)
    mensagem = f"""
## Resumo da EDA

### Qualidade dos Dados
- ✅ É uma Série Temporal
- ✅ Frequência: `{formata['frequencia']} ({formata['rotulo']})`
- ✅ {len(df)} observações mensais ({formata['inicio']} a {formata['fim']}) 
"""

    if  len(lacunas_datas) == 0:
        mensagem += f"""
- ✅ {len(lacunas_datas)} lacunas de datas
"""
    else:
        mensagem += f"""
- ❌ Esta série tem as seguintes lacunas no período {lacunas_datas}
"""
    if  nulos == 0:
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
- ⚠️ Esta série tem os seguintes péríodos repetidos {duplicatas_datas}
"""
    if len(periodos_multiplos) == 0:    
        mensagem += f"""
- ✅ {len(periodos_multiplos)} períodos com marcações múltiplas
"""
    else:
        mensagem += f"""
- ⚠️ Esta série tem os seguintes péríodos com marcações múltiplas {multiplas_detalhe}
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
        'lacunas': len(lacunas_datas),
        'lacunas_datas': list(lacunas_datas.strftime('%d/%m/%Y')),
        'duplicatas': int(df.index.duplicated().sum()),
        'duplicatas_datas': list(duplicatas_datas.strftime('%d/%m/%Y')),
        'multiplas': len(periodos_multiplos),
        'multiplas_detalhe': multiplas_detalhe,
        'nulos': int(nulos),
        'negativos': negativos,
        'zeros': zeros
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
    
def tabela_lags_significativos(interp, conf_int_val):
    """Gera tabela de lags significativos para ACF e PACF."""
    linhas = []
    for lag in range(1, 11):  # lags 1 a 10
        acf_val = acf_diff_s[lag] if lag < len(acf_diff_s) else 0
        pacf_val = pacf_diff_s[lag] if lag < len(pacf_diff_s) else 0
        acf_sig = "✅" if abs(acf_val) > conf_int_val else "—"
        pacf_sig = "✅" if abs(pacf_val) > conf_int_val else "—"
        linhas.append(f"| {lag} | {acf_val:+.3f} {acf_sig} | {pacf_val:+.3f} {pacf_sig} |")
    # Lags sazonais
    for lag in [12, 24, 36]:
        if lag < len(acf_diff_s):
            acf_val = acf_diff_s[lag]
            pacf_val = pacf_diff_s[lag]
            acf_sig = "✅" if abs(acf_val) > conf_int_val else "—"
            pacf_sig = "✅" if abs(pacf_val) > conf_int_val else "—"
            linhas.append(f"| **{lag}** (sazonal) | {acf_val:+.3f} {acf_sig} | {pacf_val:+.3f} {pacf_sig} |")
    return "\n".join(linhas)

'''
Dando os devidos créditos:
Esta função foi recebida no curso de Modelagem de Séries Temporais e Real-Time Analytics 
com Apache Spark e Databricks oferecido pela DSA - Data Science Academy
'''

# Função para testar a estacionaridade
def dsa_testa_estacionaridade(serie, window = 12, title = 'Estatísticas Móveis e Teste Dickey-Fuller'):
    """
    Função para testar a estacionaridade de uma série temporal.
    
    Parâmetros:
    - serie: pandas.Series. Série temporal a ser testada.
    - window: int. Janela para cálculo das estatísticas móveis.
    - title: str. Título para os gráficos.
    """
    # Calcula estatísticas móveis
    rolmean = serie.rolling(window = window).mean()
    rolstd = serie.rolling(window = window).std()

    # Plot das estatísticas móveis
    plt.figure(figsize = (14, 6))
    plt.plot(serie, color = 'blue', label = 'Original')
    plt.plot(rolmean, color = 'red', label = 'Média Móvel')
    plt.plot(rolstd, color = 'black', label = 'Desvio Padrão Móvel')
    plt.legend(loc = 'best')
    plt.title(title)
    plt.show(block = False)
    
    # Teste Dickey-Fuller
    print('\nResultado do Teste Dickey-Fuller:')
    dfteste = adfuller(serie, autolag = 'AIC')
    dfsaida = pd.Series(dfteste[0:4], index = ['Estatística do Teste', 
                                               'Valor-p', 
                                               'Número de Lags Consideradas', 
                                               'Número de Observações Usadas'])
    for key, value in dfteste[4].items():
        dfsaida['Valor Crítico (%s)' % key] = value
        
    print(dfsaida)
    
    # Conclusão baseada no valor-p
    if dfsaida['Valor-p'] > 0.05:
        print('\nConclusão:\nO valor-p é maior que 0.05 e, portanto, não temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente não é estacionária.')
    else:
        print('\nConclusão:\nO valor-p é menor que 0.05 e, portanto,temos evidências para rejeitar a hipótese nula.\nEssa série provavelmente é estacionária.')
        
