import networkx as nx
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
G2 = nx.read_weighted_edgelist("../../../dataset/p2p-31.e")
nx.draw_networkx(G2)
# plt.show()
ppr_source = '6'
personalization = {ppr_source: 1}
pr = nx.pagerank(
    G2, alpha=0.85, weight='weight')
# pr = 1
f = open('../../../build/output_pr_31/networkx_0.txt', 'w')

print(pr, file=f)
