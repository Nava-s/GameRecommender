# Videogame Recommendation System

## Description

This project implements a videogame recommendation system based on **Word2Vec**, using game genres as features. The system suggests similar games to those selected by the user based on genre similarities. The app is built with **Streamlit**, allowing users to interact easily with the system and receive personalized recommendations.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Pandas (for data manipulation)
  - NumPy (for numerical operations)
  - Scikit-learn (for machine learning)
  - Gensim (for Word2Vec implementation)
  - Streamlit (for creating the interactive web application)
  - Pickle (for saving and loading models)

## Dataset

The dataset used in this project is from Kaggle. It contains information about various video games, including their titles and genres. The genres are used as input for creating Word2Vec embeddings, which are then used for game recommendations.

You can access the original dataset [here](https://www.kaggle.com/datasets/arnabchaki/popular-video-games-1980-2023?resource=download/).

## How It Works

1. **Data Preprocessing**:
   - The dataset is cleaned to remove any games with missing genres.
   - The genres are split into lists for each game.
   
2. **Word2Vec Embedding**:
   - A Word2Vec model is trained using the genres of the games. The model creates vector representations for each genre.
   - An embedding vector is generated for each game by averaging the vectors of its genres.

3. **Similarity Calculation**:
   - The cosine similarity between the genre embeddings of the games is calculated to determine which games are similar.

4. **Recommendations**:
   - Given a list of games selected by the user, the system calculates the average genre similarity for the selected games and suggests other similar games based on the top N results.

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/videogame-recommender.git
cd videogame-recommender
