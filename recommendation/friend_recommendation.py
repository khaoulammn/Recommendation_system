import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Charger les relations d'amitié
friends_df = pd.read_csv("data/friends.csv")

print(friends_df.head())

# Créer un graphe non orienté
G = nx.Graph()

# Ajouter les relations d'amitié
for _, row in friends_df.iterrows():
    G.add_edge(row["user_id"], row["friend_id"])

print("Nombre d'utilisateurs :", G.number_of_nodes())
print("Nombre de relations :", G.number_of_edges())

user_id = 1
friends_of_user = list(G.neighbors(user_id))
print(f"Amis de l'utilisateur {user_id} :", friends_of_user)



plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_size=800)
plt.show()

#fonction Jaccard pour la recommandation
def recommend_friends_jaccard(G, user_id, top_n=5):
    recommendations = {}

    user_friends = set(G.neighbors(user_id))
    user_friends.add(user_id)  # éviter de se recommander soi-même

    for node in G.nodes():
        if node not in user_friends:
            neighbors_node = set(G.neighbors(node))
            intersection = len(neighbors_node & set(G.neighbors(user_id)))
            union = len(neighbors_node | set(G.neighbors(user_id)))
            if union > 0:
                jaccard_score = intersection / union
                if jaccard_score > 0:
                    recommendations[node] = jaccard_score

    sorted_recommendations = sorted(
        recommendations.items(), key=lambda x: x[1], reverse=True
    )

    return sorted_recommendations[:top_n]


user_id = 5
recommended_friends_jaccard = recommend_friends_jaccard(G, user_id)

print(f"Recommandations Jaccard pour l'utilisateur {user_id}:")
for user, score in recommended_friends_jaccard:
    print(f"Utilisateur {user} — score: {score:.2f}")


# Graphe complet
pos = nx.spring_layout(G)

# Tous les nœuds
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)

# Toutes les arêtes
nx.draw_networkx_edges(G, pos)

# Labels
nx.draw_networkx_labels(G, pos)

# Ajouter les recommandations avec une couleur différente
recommended_nodes = [user for user, score in recommended_friends_jaccard]
nx.draw_networkx_nodes(G, pos, nodelist=recommended_nodes, node_color='orange', node_size=800)

plt.title(f"Recommandations d'amis pour l'utilisateur {user_id} (orange)")
plt.show()
