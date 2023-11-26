import networkx as nx

for i in range(339):
    graph = nx.read_graphml("graph_"+str(i)+".graphml")
    nx.write_gml(graph, "../GML/graph_"+str(i)+".gml")