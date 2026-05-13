import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
import os
import zipfile
import urllib.request

@st.cache_data
def download_movielens():
    """Baixa dataset MovieLens 100K"""
    dest = 'ml-100k'
    if os.path.exists(dest):
        return
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = 'ml-100k.zip'
    if not os.path.exists(zip_path):
        st.info('ðŸ“¥ Baixando dataset MovieLens 100K (~6MB)...')
        urllib.request.urlretrieve(url, zip_path)
    st.info('ðŸ“¦ Extraindo arquivos...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(zip_path)
    st.success('âœ… Dataset pronto!')

@st.cache_data
def load_data():
    """Carrega ratings e filmes"""
    download_movielens()
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', 
                         names=['userId', 'movieId', 'rating', 'timestamp'],
                         usecols=[0, 1, 2])
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='ISO-8859-1',
                        names=['movieId', 'title'] + ['genre']*19,
                        usecols=[0, 1])
    return ratings, movies

@st.cache_data
def compute_rules():
    """Calcula regras de associaÃ§Ã£o com Apriori"""
    ratings, movies = load_data()

    # Filtra ratings altos (>=4)
    high_ratings = ratings[ratings['rating'] >= 4]

    # Cria transaÃ§Ãµes (filmes por usuÃ¡rio)
    user_movies = high_ratings.groupby('userId')['movieId'].apply(list).tolist()

    # One-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(user_movies).transform(user_movies)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_, dtype=bool)

    # Apriori
    itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)

    # Regras de associaÃ§Ã£o
    rules = association_rules(itemsets, metric='confidence', min_threshold=0.3)
    rules = rules[rules['lift'] > 1]

    # Mapear IDs para tÃ­tulos
    movie_to_id = dict(zip(movies['title'], movies['movieId']))
    id_to_movie = {v: k for k, v in movie_to_id.items()}
    movie_list = sorted(movies['title'].tolist())

    return rules, movie_to_id, id_to_movie, movie_list

# ===== STREAMLIT APP =====
st.set_page_config(page_title="Recomendador MovieLens", layout="wide", initial_sidebar_state="expanded")

st.title("ðŸŽ¬ Recomendador de Filmes com Apriori & Regras de AssociaÃ§Ã£o")
st.markdown("**Baseado em:** MovieLens 100K | **MÃ©todo:** Apriori + Association Rules | **MÃ©trica:** Confidence & Lift")

# Carregar dados
rules, movie_to_id, id_to_movie, movie_list = compute_rules()

# ===== SIDEBAR: EXEMPLOS PRÃ‰-CONFIGURADOS =====
st.sidebar.markdown("## ðŸ“‹ Exemplos PrÃ©-configurados")

def set_example(movies):
    st.session_state['movie1'] = movies[0] if movies[0] in movie_list else movie_list[0]
    st.session_state['movie2'] = movies[1] if movies[1] in movie_list else movie_list[0]
    st.session_state['movie3'] = movies[2] if movies[2] in movie_list else movie_list[0]
    st.rerun()

if st.sidebar.button("ðŸŽ¥ ClÃ¡ssicos Famosos"):
    set_example(["Toy Story (1995)", "Star Wars (1977)", "Pulp Fiction (1994)"])

if st.sidebar.button("ðŸ’¥ Filmes de AÃ§Ã£o"):
    set_example(["Terminator 2: Judgment Day (1991)", "Matrix, The (1999)", "Raiders of the Lost Ark (1981)"])

if st.sidebar.button("ðŸ’• Romances"):
    set_example(["Pretty Woman (1990)", "When Harry Met Sally... (1989)", "Sleepless in Seattle (1993)"])

st.sidebar.markdown("---")
st.sidebar.info(
    "ðŸ’¡ **Como funciona:**\n"
    "1. Selecione 3 filmes (ou use um dos exemplos)\n"
    "2. Clique 'Gerar RecomendaÃ§Ãµes'\n"
    "3. Veja os filmes recomendados com **confianÃ§a** e **lift**"
)

# ===== MAIN: SELEÃ‡ÃƒO DE FILMES =====
st.markdown("## ðŸŽ¯ Selecione 3 filmes que vocÃª gosta:")

col1, col2, col3 = st.columns(3)

with col1:
    movie1 = st.selectbox("ðŸŽ¬ Filme 1:", movie_list, key='movie1', label_visibility="collapsed")

with col2:
    movie2 = st.selectbox("ðŸŽ¬ Filme 2:", movie_list, key='movie2', label_visibility="collapsed")

with col3:
    movie3 = st.selectbox("ðŸŽ¬ Filme 3:", movie_list, key='movie3', label_visibility="collapsed")

selected_movies = [movie1, movie2, movie3]

# ===== GERAR RECOMENDAÃ‡Ã•ES =====
if st.button("ðŸ” Gerar RecomendaÃ§Ãµes", type="primary", use_container_width=True):
    with st.spinner("â³ Analisando padrÃµes..."):
        selected_ids = [movie_to_id.get(m) for m in selected_movies if m in movie_to_id]

        if len(selected_ids) == 0:
            st.warning("âš ï¸ Selecione pelo menos um filme vÃ¡lido.")
        else:
            S = frozenset(selected_ids)
            recs = []

            # Encontrar regras onde antecedente estÃ¡ em S
            for _, rule in rules.iterrows():
                antecedents = rule['antecedents']
                if antecedents.issubset(S):
                    consequents = rule['consequents']
                    confidence = rule['confidence']
                    lift_ = rule['lift']

                    for con_id in consequents:
                        if con_id not in S:
                            title = id_to_movie[con_id]
                            recs.append({
                                'movie_id': con_id,
                                'titulo': title,
                                'confianca': confidence,
                                'lift': lift_
                            })

            # Remove duplicatas
            rec_dict = {}
            for item in recs:
                con_id = item['movie_id']
                if con_id not in rec_dict or item['confianca'] > rec_dict[con_id]['confianca']:
                    rec_dict[con_id] = item

            rec_list = sorted(rec_dict.values(), key=lambda x: x['confianca'], reverse=True)[:15]

            if rec_list:
                st.success(f"âœ… Encontradas **{len(rec_list)}** recomendaÃ§Ãµes!")

                # ===== TABELA DE RESULTADOS =====
                st.markdown("### ðŸ“Š Resultados")
                df_rec = pd.DataFrame(rec_list)
                df_rec['confianca_pct'] = (df_rec['confianca'] * 100).round(2).astype(str) + '%'
                df_rec['lift_rounded'] = df_rec['lift'].round(2)

                display_df = df_rec[['titulo', 'confianca_pct', 'lift_rounded']].copy()
                display_df.columns = ['ðŸ“½ï¸ TÃ­tulo do Filme', 'ðŸ“Š ConfianÃ§a', 'ðŸ“ˆ Lift']
                display_df.index = display_df.index + 1

                st.dataframe(display_df, use_container_width=True, height=400)

                # ===== GRÃFICOS =====
                st.markdown("### ðŸ“ˆ VisualizaÃ§Ãµes")

                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    fig_conf = px.bar(
                        df_rec.head(10), 
                        x='titulo', 
                        y='confianca',
                        title="ConfianÃ§a das RecomendaÃ§Ãµes (Top 10)",
                        labels={'confianca': 'ConfianÃ§a', 'titulo': 'Filme'},
                        color='confianca',
                        color_continuous_scale='Viridis',
                        text='confianca'
                    )
                    fig_conf.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                    fig_conf.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
                    st.plotly_chart(fig_conf, use_container_width=True)

                with col_chart2:
                    fig_lift = px.scatter(
                        df_rec.head(10),
                        x='confianca',
                        y='lift',
                        size='lift',
                        hover_name='titulo',
                        title="ConfianÃ§a vs Lift",
                        labels={'confianca': 'ConfianÃ§a', 'lift': 'Lift'},
                        color='confianca',
                        color_continuous_scale='Plasma',
                        size_max=50
                    )
                    fig_lift.update_layout(height=500)
                    st.plotly_chart(fig_lift, use_container_width=True)

                # ===== EXPLICAÃ‡Ã•ES =====
                st.markdown("### ðŸ’¡ O que significa?")

                col_exp1, col_exp2, col_exp3 = st.columns(3)

                with col_exp1:
                    st.metric("ðŸ“Š ConfianÃ§a", "P(Y|X)", "Probabilidade de recomendar Y dado X")

                with col_exp2:
                    st.metric("ðŸ“ˆ Lift", "ForÃ§a da relaÃ§Ã£o", "Quanto melhor que aleatÃ³rio")

                with col_exp3:
                    st.metric("ðŸŽ¯ Top RecomendaÃ§Ã£o", rec_list[0]['titulo'][:20], f"ConfianÃ§a: {rec_list[0]['confianca']:.1%}")

            else:
                st.info("â„¹ï¸ Nenhuma recomendaÃ§Ã£o encontrada para essa combinaÃ§Ã£o. Tente outros filmes!")

st.markdown("---")
st.caption(
    "ðŸ”¬ **Tecnologia:** Apriori Algorithm | Market Basket Analysis\n"
    "ðŸ“Š **Dataset:** MovieLens 100K (ratings >= 4 stars)\n"
    "âš™ï¸ **Regras:** Min Support=1% | Min Confidence=30% | Min Lift=1.0"
)