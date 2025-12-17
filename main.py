from recommendation.friend_recommendation import load_graph, recommend_friends_jaccard, visualize_graph
from recommendation.post_recommendation import load_posts_likes, create_user_post_matrix, compute_user_similarity, recommend_posts


#  Charger les données

G = load_graph("data/friends.csv")
posts_df, likes_df = load_posts_likes("data/posts.csv", "data/likes.csv")

#  Définir un utilisateur de test

user_id = 5

#  Recommandation d'amis

recommended_friends = recommend_friends_jaccard(G, user_id)
print(f"Recommandations d’amis pour l’utilisateur {user_id}:")
for user, score in recommended_friends:
    print(f"Utilisateur {user} — Score: {score:.2f}")

# Visualisation du graphe
# Afficher le graphe global avant recommandations
visualize_graph(G, show_global=True)

# Afficher le graphe avec recommandations
recommended_nodes = [user for user, score in recommended_friends]
visualize_graph(G, recommended_nodes=recommended_nodes, user_id=user_id)

#  Recommandation de posts

user_post_matrix = create_user_post_matrix(posts_df, likes_df)
user_sim_matrix = compute_user_similarity(user_post_matrix)
recommended_posts = recommend_posts(user_id, user_post_matrix, user_sim_matrix)

print(f"\nPosts recommandés pour l’utilisateur {user_id}:")
for post_id, score in recommended_posts:
    content = posts_df.loc[posts_df['post_id'] == post_id, 'content'].values[0]
    print(f"Post {post_id}: '{content}' — Score {score:.2f}")
