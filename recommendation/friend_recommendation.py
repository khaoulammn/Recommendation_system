import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_graph(file_path):
    
    #Charge le fichier friends.csv et crée un graphe NetworkX.
    #Chaque utilisateur est un noeud, chaque relation d'amitié est une arête.
    
    df = pd.read_csv(file_path)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["user_id"], row["friend_id"])
    return G

def recommend_friends_jaccard(G, user_id, top_n=5, threshold=0.2):
    
    #Recommande des amis pour un utilisateur en utilisant le coefficient de Jaccard.
    
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
                if jaccard_score > threshold:
                    recommendations[node] = jaccard_score

    # Trier par score décroissant et retourner le top_n
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations[:top_n]

def visualize_graph(G, recommended_nodes=[], user_id=None, show_global=False):
    """
    Affiche le graphe.
    Si show_global=True, affiche le graphe complet sans recommandations.
    Si recommended_nodes est fourni, les met en orange.
    """
    pos = nx.spring_layout(G)
    
    # Si show_global = True, afficher le graphe complet initial
    if show_global:
        plt.figure(figsize=(8,6))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        plt.title("Graphe global des relations")
        plt.show()
    
    # Graphe avec recommandations en orange (si fourni)
    if recommended_nodes:
        plt.figure(figsize=(8,6))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_nodes(G, pos, nodelist=recommended_nodes, node_color='orange', node_size=800)
        if user_id:
            plt.title(f"Recommandations d'amis pour l'utilisateur {user_id} (orange)")
        plt.show()
