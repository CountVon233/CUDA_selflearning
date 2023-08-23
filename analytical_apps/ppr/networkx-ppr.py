import networkx as nx
import matplotlib.pyplot as plt

fig = plt.figure()
G2 = nx.read_weighted_edgelist("../../../dataset/p2p-31.e")
ppr_source = '2'
personalization = {ppr_source: 1}
pr = nx.pagerank(G2, alpha=0.85, personalization=personalization, max_iter=1000, tol=1e-6)
# pr = nx.pagerank(G2, alpha=0.85, personalization=None)
# pr = 1
f = open('../../../build/output_ppr_auto_31/networkx_0.txt', 'w')

print(pr, file=f)
