# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:28:46 2016

@author: jason
"""
import networkx as nx
import matplotlib.pyplot as plt
G=nx.karate_club_graph()
nx.draw(G, with_labels=True, node_color="lightblue",edge_color="gray")
plt.savefig("karate_graph.pdf")

from scipy.stats import bernoulli
N = 20
p = 0.2

def er_graph(N, p):
    """
        Generate an er graph.
    """
    # create empty graph
    # add all N in the graph
    # loop over all pairs of nodes
        # add an edge with prob p
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if bernoulli.rvs(p=p) and node1<node2:
                G.add_edge(node1,node2)
    return G
    
nx.draw(er_graph(50,0.08),node_size=40,node_color="gray")
plt.savefig("er1.pdf")

def plot_degree_distribution(G):
    plt.hist(list(G.degree().values()),histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree distribution")
    
G1 = er_graph(500,0.08)
plot_degree_distribution(G1)
G2 = er_graph(500,0.08)
plot_degree_distribution(G2)
G3 = er_graph(500,0.08)
plot_degree_distribution(G3)
plt.savefig("hist3.pdf")
    
import numpy as np
A1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv",delimiter=",")
A2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv",delimiter=",")
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
    print("Number of nodes: %d" %G.number_of_nodes())
    print("Number of edges: %d" %G.number_of_edges())
    print("Mean degrees: %2f" %np.mean(list(G.degree().values())))

plot_degree_distribution(G1)
plot_degree_distribution(G2)
plt.savefig("village_hist.pdf") 
    

gen = nx.connected_component_subgraphs(G1)
G1_LCC = max(nx.connected_component_subgraphs(G1), key=len)
G2_LCC = max(nx.connected_component_subgraphs(G2), key=len)

G1_LCC.number_of_nodes()/G1.number_of_nodes()
G2_LCC.number_of_nodes()/G2.number_of_nodes()

plt.figure()
nx.draw(G1_LCC, node_color="red", edge_color="gray",node_size=20)
plt.savefig("village1.pdf")
nx.draw(G2_LCC, node_color="green", edge_color="gray",node_size=20)
plt.savefig("village2.pdf")