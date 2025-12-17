import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Chargement des posts
posts_df = pd.read_csv("data/posts.csv")
likes_df = pd.read_csv("data/likes.csv")

print(posts_df.head())
print(likes_df.head())

# Liste des utilisateurs et posts
users = posts_df['user_id'].unique()
posts = posts_df['post_id'].unique()

# Initialiser une matrice - utilisateur x post
user_post_matrix = pd.DataFrame(0, index=users, columns=posts)

# Remplir la matrice avec les likes
for _, row in likes_df.iterrows():
    user_post_matrix.at[row['user_id'], row['post_id']] = 1

print(user_post_matrix)

# Similarité entre tous les utilisateurs
user_sim_matrix = pd.DataFrame(
    cosine_similarity(user_post_matrix),
    index=user_post_matrix.index,
    columns=user_post_matrix.index
)

print(user_sim_matrix)

#Recommander des posts pour un utilisateur
def recommend_posts(user_id, user_post_matrix, user_sim_matrix, top_n=5):
    # Similarité avec les autres utilisateurs
    sim_scores = user_sim_matrix[user_id]

    # Pondération des posts des autres utilisateurs par leur similarité
    scores = user_post_matrix.T.dot(sim_scores)

    # Exclure les posts déjà likés
    liked_posts = set(user_post_matrix.columns[user_post_matrix.loc[user_id] > 0])
    
    recommended = [
    (post, score)
    for post, score in scores.items()
    if post not in liked_posts and score > 0
    ]

    # Trier par score décroissant
    recommended.sort(key=lambda x: x[1], reverse=True)

    return recommended[:top_n]


user_id = 1
recommended_posts = recommend_posts(user_id, user_post_matrix, user_sim_matrix)

print(f"Posts recommandés pour l'utilisateur {user_id}:")
for post_id, score in recommended_posts:
    content = posts_df.loc[posts_df['post_id']==post_id, 'content'].values[0]
    print(f"Post {post_id}: '{content}' — Score {score:.2f}")
