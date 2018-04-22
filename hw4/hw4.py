import matplotlib
matplotlib.use('Agg')

import itertools
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from statistics import mean, median

data = pd.read_csv('casts.csv', sep=';', header=None)

G = nx.Graph()
lines = len(data)
i = 0

films = {}

while i < lines:
    if data[2][i] == 's a':
        i += 1
        continue

    if data[0][i] not in films:
        films[data[0][i]] = set()

    films[data[0][i]].add(data[2][i])

    if data[2][i] not in G:
        G.add_node(data[2][i])

    i += 1

for index in films:
    film = films[index]
    for pair in itertools.combinations(list(film), 2):
        G.add_edge(pair[0], pair[1])

# nx.draw_kamada_kawai(G, with_labels=True)
# plt.savefig("graph.png")

top_n = 5


# Basic statistic
nodes_number = len(G.nodes())
edges_number = len(G.edges())
density = nx.density(G)
components_number = len(list(nx.connected_components(G)))

print('Number of nodes: {}'.format(nodes_number))
print('Number of edges: {}'.format(edges_number))
print('Density of the graph: {}'.format(density))
print('Number of components: {}'.format(components_number))
print()


# Centralities
centralities = [
    ('Degree', nx.degree_centrality),
    # ('Closeness', nx.closeness_centrality),
    # ('Betweeness', nx.betweenness_centrality),
    # ('Eigenvector', nx.eigenvector_centrality)
]

for centrality in centralities:
    print('{} centrality: (top {})'.format(centrality[0], top_n))
    c = [(k, centrality[1](G)[k]) for k in centrality[1](G)]
    for entry in c:
        G.node[entry[0]]['{} centrality'.format(centrality[0])] = entry[1]

    for entry in sorted(c, key=lambda x: -x[1])[:top_n]:
        print('{}: {}'.format(entry[0], entry[1]))
    print()


# Kevin Bacon numbers
path = nx.single_source_shortest_path(G, 'Kevin Bacon')
for index in path:
    if isinstance(index, float):
        continue
    G.node[index]['Kevin Bacon'] = len(path[index])-1

print("Kevin Bacon distance")
KB_distances_counts = Counter([x[1] for x in list(G.nodes.data('Kevin Bacon'))])
for i in range(8):
    print('{}: {}'.format(i, KB_distances_counts[i]))
print('{}: {}'.format('infinity', KB_distances_counts[None]))
print()


# Clustering
communities = {node: cid+1 for cid, community in enumerate(nx.algorithms.community.k_clique_communities(G, 3)) for node in community}

for index in communities:
    G.node[index]['community'] = communities[index]

communities_sizes = sorted(Counter(communities.values()).values(), reverse=True)
print('Sizes of top {} communities: {}'.format(top_n, communities_sizes[:top_n]))
print('Average community size: {}'.format(mean(communities_sizes)))
print('Median community size: {}'.format(median(communities_sizes)))

# plt.figure(2, figsize=(12, 12))
# nx.draw_kamada_kawai(G, labels={v: str(v) for v in G}, cmap=plt.get_cmap("rainbow"), node_color=[communities[v] if v in communities else 0 for v in G])
# plt.savefig("communities.png")


# Export
nx.write_gexf(G, "graph.gexf")

