import networkx as nx
import matplotlib.pyplot as plt
import collections
import scipy
import numpy as np
import sys
from numpy import genfromtxt

def choose_network(graph_n):
    graph = None

    if graph_n == 1:
        # Ler Word Adjancencies Network
        graph = nx.read_gml("networks_data/adjnoun.gml")
        nx.draw(graph, with_labels=1)
    elif graph_n == 2:
        # Ler Italian Gangs Network
        graph_data = genfromtxt('networks_data/italian_gangs.csv', delimiter=',')
        adjacency = graph_data[1:,1:]
        rows, cols = np.where(adjacency == 1)
        edges = zip(rows.tolist(), cols.tolist())
        graph = nx.Graph()
        graph.add_edges_from(edges)

        countries_data = genfromtxt('networks_data/italian_gangs_attr.csv', delimiter=',')
        countries = countries_data[1:,-1]
        countries_dict = {}
        for i in range(len(countries)):
            countries_dict[i] = {"country" : int(countries[i])}
        nx.set_node_attributes(graph, countries_dict)
        nx.draw(graph, labels=nx.get_node_attributes(graph, "country"), with_labels=1)
    elif graph_n == 3:
        # Ler JUnit Framework Dependencies Network
        graph = nx.read_edgelist('networks_data/junit.txt', comments='#')
        nx.draw(graph, with_labels=1)
    else:
        print('Choose a graph between 1 and 3!')
        print('     1 - Word Adjancecies')
        print('     2 - Italian Gangs')
        print('     3 - JUnit Dependecies')
        return

    print(nx.info(graph))
    plt.show()

    print("<k> : " + str(avg_degree(graph)))
    print("A average path length é: ", nx.average_shortest_path_length(graph))
    # print("O meu cluster é : ", cluster_coefficient_of_node(graph, 2))
    # print("O real cluster é : ", nx.clustering(graph, 2))
    print("The FAKE average cluster coefficient is: ", avg_cluster_coefficient(graph))
    print("The REAL average cluster coefficient is: ", nx.average_clustering(graph))
    degree_dist(graph)



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