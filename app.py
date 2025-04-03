import streamlit as st
import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


# Carica il dataset e il modello pre-addestrato
@st.cache_data
def load_data():
    df = pd.read_csv("processed_games.csv")
    return df

@st.cache_resource
def load_model():
    with open("word2vec_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# Funzione per calcolare l'embedding medio di un insieme di generi
def genre_embedding_vector(genre_list, model):
    vectors = [model.wv[word] for word in genre_list if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Pre-calcola i vettori per tutti i giochi
df["Genre_Embedding"] = df["Genres"].apply(lambda x: genre_embedding_vector(eval(x), model))

# Matrice di similarità
genre_embeddings = np.stack(df["Genre_Embedding"].values)
genre_sim_embed = cosine_similarity(genre_embeddings, genre_embeddings)

# Funzione di raccomandazione
def recommend_games(selected_titles, df, genre_sim_embed, top_n=5):
    indices = [df.index[df["Title"] == title][0] for title in selected_titles if title in df["Title"].values]
    avg_sim_scores = np.mean([genre_sim_embed[idx] for idx in indices], axis=0)
    sim_scores = list(enumerate(avg_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Rimuove i giochi già selezionati e prende i top_n suggeriti
    game_indices = [i[0] for i in sim_scores if df["Title"].iloc[i[0]] not in selected_titles][:top_n]
    return df["Title"].iloc[game_indices].tolist()

# ---- INTERFACCIA STREAMLIT ----
st.title("🎮 Retro Games Recommender")
st.subheader("Seleziona un gioco e ottieni 5 suggerimenti!")

# Selezione dei giochi
selected_games = st.multiselect("Scegli i tuoi giochi preferiti:", df["Title"].unique(), max_selections=3)

# Bottone per generare raccomandazioni
if st.button("Suggerisci Giochi"):
    if selected_games:
        recommendations = recommend_games(selected_games, df, genre_sim_embed)
        st.subheader("🎯 Ti potrebbero piacere anche:")
        for game in recommendations:
            st.write(f"- {game}")
    else:
        st.warning("⚠️ Seleziona un gioco!")

# Footer
st.markdown("---")
st.markdown("📌 Progetto sviluppato da **Antonio Proietti**")
