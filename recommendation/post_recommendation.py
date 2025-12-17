# post_recommendation.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_posts_likes(posts_file, likes_file):
    
    #Charge les fichiers posts.csv et likes.csv et retourne deux DataFrames.
    
    posts_df = pd.read_csv(posts_file)
    likes_df = pd.read_csv(likes_file)
    return posts_df, likes_df

def create_user_post_matrix(posts_df, likes_df):
    
    #Crée une matrice utilisateur x post avec 1 si l'utilisateur a liké le post, sinon 0.
    
    users = posts_df['user_id'].unique()
    posts = posts_df['post_id'].unique()
    matrix = pd.DataFrame(0, index=users, columns=posts)
    for _, row in likes_df.iterrows():
        matrix.at[row['user_id'], row['post_id']] = 1
    return matrix

def compute_user_similarity(user_post_matrix):
    
    #Calcule la similarité cosinus entre tous les utilisateurs.
    
    return pd.DataFrame(
        cosine_similarity(user_post_matrix),
        index=user_post_matrix.index,
        columns=user_post_matrix.index
    )

def recommend_posts(user_id, user_post_matrix, user_sim_matrix, top_n=5):
    
    #Recommande des posts pour un utilisateur donné en filtrant ceux déjà likés.
    
    sim_scores = user_sim_matrix[user_id]
    scores = user_post_matrix.T.dot(sim_scores)
    liked_posts = set(user_post_matrix.columns[user_post_matrix.loc[user_id] > 0])
    recommended = [(post, score) for post, score in scores.items() if post not in liked_posts and score > 0]
    recommended.sort(key=lambda x: x[1], reverse=True)
    return recommended[:top_n]
