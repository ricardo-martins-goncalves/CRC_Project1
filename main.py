import networkx as nx
import matplotlib.pyplot as plt
import collections
import scipy

#graph = nx.read_gml("C:/Users/Pedro Guerra/Desktop/Curso/Network Science/lesmis.gml", label = 'id')
#print(nx.info(graph))

graph = nx.read_adjlist("C:/Users/Casa/Desktop/porgat.txt", nodetype=str)
print(nx.info(graph))

"""
print("deu 2")
nx.draw(graph, with_labels=1)
plt.show()

print("deu")
"""

def degree_dist(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(G)
    plt.axis("off")
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.show()

def avg_degree(G):
    sum_of_edges = 0
    number_of_nodes = nx.number_of_nodes(G)
    print('ALKNFALKSFJ' + str(number_of_nodes))
    edges_ind = 0
    while edges_ind < number_of_nodes:
        sum_of_edges += G.degree[edges_ind]
        edges_ind += 1
    avg_degree = sum_of_edges / number_of_nodes
    print("O Degree médio é: ", avg_degree)

def neighbors(G, node_id):
    list_of_neighbors = [n for n in G.neighbors(node_id)]
    return list_of_neighbors

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def cluster_coefficient_of_node(G, node_id):
    neighbors_of_node = neighbors(G, node_id)
    pairs_of_neighbors_connected = []
    for neighbor in neighbors_of_node:
        neighbors_of_neighbor = neighbors(G, neighbor)
        neighb_among_neighb = intersection(neighbors_of_node, neighbors_of_neighbor)
        for neighbor_of_neighbor in neighb_among_neighb:
            if (neighbor, neighbor_of_neighbor) and (neighbor_of_neighbor, neighbor) not in pairs_of_neighbors_connected:
                pairs_of_neighbors_connected.append((neighbor, neighbor_of_neighbor))
    number_of_neighbors = len(neighbors_of_node)
    if pairs_of_neighbors_connected:
        number_of_connections_between_neighb = len(pairs_of_neighbors_connected)
    else:
        number_of_connections_between_neighb = 0
    if number_of_neighbors == 0 or number_of_neighbors == 1:
        cluster_coefficient = 0
    else:
        cluster_coefficient = number_of_connections_between_neighb / ((number_of_neighbors * (number_of_neighbors - 1)) / 2)
    return cluster_coefficient

def avg_cluster_coefficient(G):
    number_of_nodes = nx.number_of_nodes(G)
    sum_of_cluster_coefficients = 0.0
    for node_id in range(1, number_of_nodes + 1):
        sum_of_cluster_coefficients += cluster_coefficient_of_node(G, node_id)
        node_id += 1
    avg_cluster_coefficient = sum_of_cluster_coefficients / number_of_nodes
    return avg_cluster_coefficient

print(graph.degree[1])
# avg_degree(graph)
print("A average path length é: ", nx.average_shortest_path_length(graph))
print("O meu cluster é : ", cluster_coefficient_of_node(graph, 2))
print("O real cluster é : ", nx.clustering(graph, 2))
#print("The average cluster coefficient is: ", avg_cluster_coefficient(graph))
# degree_dist(graph)

