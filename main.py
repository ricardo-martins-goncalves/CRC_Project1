import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
import scipy
import numpy as np
import sys
import community as community_louvain
from numpy import genfromtxt

def choose_network(graph_n):
    graph = None

    if graph_n == 1:
        # Ler Word Adjancencies Network

        graph = read_graph('gml', 'networks_data/adjnoun.gml')
        draw_graph(graph)

    elif graph_n == 2:
        # Ler Italian Gangs Network

        graph = read_graph('c_edgelist', 'networks_data/italian_gangs.csv', attr_path='networks_data/italian_gangs_attr.csv')
        draw_graph(graph, nx.get_node_attributes(graph, "country"))

    elif graph_n == 3:
        # Ler JUnit Framework Dependencies Network

        graph = read_graph('s_edgelist', 'networks_data/junit.txt', comments='#')
        draw_graph(graph)

    else:
        print('Choose a graph between 1 and 3!')
        print('     1 - Word Adjancecies')
        print('     2 - Italian Gangs')
        print('     3 - JUnit Dependecies')
        return

    print(nx.info(graph))
    plt.show()
    show_communities(graph)

    print("<k> : " + str(avg_degree(graph)))
    print("The Average Path Length is: ", avg_shortest_path_length(graph))
    # print("O meu cluster é : ", cluster_coefficient_of_node(graph, 2))
    # print("O real cluster é : ", nx.clustering(graph, 2))
    print("The FAKE average cluster coefficient is: ", avg_cluster_coefficient(graph))
    print("The REAL average cluster coefficient is: ", nx.average_clustering(graph))
    print("Betweeness Centrality is: ", nx.betweenness_centrality(graph))
    # print("Closeness Centrality is: ", nx.closeness_centrality(graph))
    # print("Degree Centrality is: ", nx.degree_centrality(graph))
    degree_dist(graph)

def read_graph(g_type, path, attr_path=None, comments='#'):
    graph = None
    if g_type == "gml":
        graph = nx.read_gml(path)

    elif g_type == "s_edgelist":
        graph = nx.read_edgelist(path, comments=comments)

    elif g_type == "c_edgelist":
        graph_data = genfromtxt(path, delimiter=',')
        adjacency = graph_data[1:,1:]
        rows, cols = np.where(adjacency == 1)
        edges = zip(rows.tolist(), cols.tolist())
        graph = nx.Graph()
        graph.add_edges_from(edges)

        countries_data = genfromtxt(attr_path, delimiter=',')
        countries = countries_data[1:,-1]
        countries_dict = {}
        for i in range(len(countries)):
            countries_dict[i] = {"country" : int(countries[i])}
        nx.set_node_attributes(graph, countries_dict)
    
    return graph

def draw_graph(graph, labels=None):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=1, labels=labels)

def show_communities(G):
    partition = community_louvain.best_partition(G)
    pos = nx.spring_layout(G)

    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=200, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "country"))
    plt.show()

def avg_shortest_path_length(G):
    if nx.connected_components(G) == 1:
        return nx.average_shortest_path_length(G)
    else:
        components_apl = []
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            components_apl.append(nx.average_shortest_path_length(C))
        return components_apl

def degree_dist(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    number_of_nodes = nx.number_of_nodes(G)
    for i, value in degreeCount.items():
        degreeCount[i] = value / number_of_nodes
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(deg, cnt, color="b")

    plt.title("Degree Distribution")
    plt.ylabel("P(k)")
    plt.xlabel("k")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(deg)

    # draw graph in inset
    # plt.axes([0.4, 0.4, 0.5, 0.5])
    # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    # pos = nx.spring_layout(G)
    # plt.axis("off")
    # nx.draw_networkx_nodes(G, pos, node_size=20)
    # nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.show()

def avg_degree(G):
    sum_of_edges = 0
    number_of_nodes = nx.number_of_nodes(G)
    for value in dict(G.degree).values():
        sum_of_edges += value
    avg_degree = sum_of_edges / number_of_nodes
    return avg_degree

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
    for node in list(G.nodes):
        sum_of_cluster_coefficients += cluster_coefficient_of_node(G, node)
    avg_cluster_coefficient = sum_of_cluster_coefficients / number_of_nodes
    return avg_cluster_coefficient




# main
choose_network(0 if len(sys.argv) != 2 else int(sys.argv[1]))