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
    """Computes properties of the graph with the given id
    
    graph_n -- graph identifier
    """
    graph = None

    if graph_n == 1:
        # Ler Italian Gangs Network

        graph = read_graph('c_edgelist', 'networks_data/italian_gangs.csv', attr_path='networks_data/italian_gangs_attr.csv')
        # draw_graph(graph, nx.get_node_attributes(graph, "country"))
        draw_graph(graph)

    elif graph_n == 2:
        # Ler JUnit Framework Dependencies Network

        graph = read_graph('s_edgelist', 'networks_data/junit.txt', comments='#')
        draw_graph(graph)

    else:
        print('Choose a graph between 1 and 2!')
        print('     1 - Italian Gangs')
        print('     2 - JUnit Dependecies')
        return

    print(nx.info(graph))
    plt.show()

    print("<k> : " + str(avg_degree(graph)))
    print("The Average Path Length is: ", avg_shortest_path_length(graph))
    sorted_centrality = {k: v for k, v in sorted(nx.degree_centrality(graph).items(), key=lambda item: item[1], reverse=True)}
    print("The average cluster coefficient is (NetworkX): ", nx.average_clustering(graph))
    print("Betweeness Centrality is: ", sorted_centrality)
    print("Degree Centrality is: ", sorted_centrality)

    if not graph.is_directed():
        print("The average cluster coefficient is (ours): ", avg_cluster_coefficient(graph))
        show_communities(graph)
        degree_dist(graph)
    else:
        degree_dist(graph, in_degree=True, title="In-Degree Distribution")
        degree_dist(graph, out_degree=True, title="Out-Degree Distribution")

def read_graph(g_type, path, attr_path=None, comments='#'):
    """Reads graph from a file. 

    g_type -- graph type
    path -- absolute or relative path to file with nodes and edges data
    attr_path -- absolute or relative path to file with attribute data
    comments -- prefix that indicates the start of a comment line
    """
    graph = None

    if g_type == "s_edgelist":
        graph = nx.read_edgelist(path, comments=comments, create_using=nx.DiGraph)

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
    """Draw graph

    graph -- target graph
    labels -- l
    """
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=1, labels=labels)

def show_communities(G):
    """Compute and plot Louvain Communities

    G -- target graph
    """
    partition = community_louvain.best_partition(G)
    pos = nx.spring_layout(G)

    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=200, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "country"))
    plt.show()

def avg_shortest_path_length(G):
    """Given a graph computes its average shortest path length
    Only for undirected graphs

    G -- target Graph
    """
    if not G.is_directed():
        if nx.connected_components(G) == 1:
            return nx.average_shortest_path_length(G)
        else:
            components_apl = []
            for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
                components_apl.append(nx.average_shortest_path_length(C))
            return components_apl
    elif G.is_directed() and nx.is_weakly_connected(G):
        components_apl = []
        for C in (G.subgraph(c).copy() for c in nx.weakly_connected_components(G)):
            components_apl.append(nx.average_shortest_path_length(C))
        return components_apl
    else:
        None

def degree_dist(G, in_degree=False, out_degree=False, title="Degree Distribution"):
    """Computes degree distribution of a given graph and a given type of degree, if none of the flags are used then a degree distribution for undirected graph is computed 

    G -- target graph
    in_degree -- flag to compute in degree distribution
    out_degree -- flag to compute out degree distribution
    title -- degree distribution title
    """
    color = ""
    if in_degree:
        degree_sequence = sorted([d for n, d in G.in_degree()], reverse=True)
        color = "b"
    elif out_degree:
        degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True)
        color = "g"
    else:
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        color = "r"
    degree_count = collections.Counter(degree_sequence)
    number_of_nodes = nx.number_of_nodes(G)
    for i, value in degree_count.items():
        degree_count[i] = value / number_of_nodes
    deg, cnt = zip(*degree_count.items())
    print(degree_count.items())
    plt.figure(figsize=(12, 8))
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(deg, cnt, color=color)

    plt.title(title)
    plt.ylabel("P(k)")
    plt.xlabel("k")
    if in_degree or out_degree:
        plt.xlim(xmin=0.1, right=100)

    plt.show()

def avg_degree(G):
    """Compute average degree of a given graph

    G -- target graph
    """
    sum_of_edges = 0
    number_of_nodes = nx.number_of_nodes(G)
    for value in dict(G.degree).values():
        sum_of_edges += value
    avg_degree = sum_of_edges / number_of_nodes
    return avg_degree

def neighbors(G, node_id):
    """Given a graph and a node id returns the node neighbors

    G -- target graph
    node_id -- id of a node in G
    """
    list_of_neighbors = [n for n in G.neighbors(node_id)]
    return list_of_neighbors

def intersection(lst1, lst2):
    """Given two lists returns a list with the elements present on both lists

    lst1 -- first list of values
    lst2 -- second list of values
    """ 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def cluster_coefficient_of_node(G, node_id):
    """Computes the clustering coefficient of a given node

    G -- graph
    node_id -- id of a node in G
    """
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
    """Computes the average clustering coeffiecent of the given graph. Only works for undirected graphs

    G -- graph
    """
    number_of_nodes = nx.number_of_nodes(G)
    sum_of_cluster_coefficients = 0.0
    for node in list(G.nodes):
        sum_of_cluster_coefficients += cluster_coefficient_of_node(G, node)
    avg_cluster_coefficient = sum_of_cluster_coefficients / number_of_nodes
    return avg_cluster_coefficient


# main
choose_network(0 if len(sys.argv) != 2 else int(sys.argv[1]))