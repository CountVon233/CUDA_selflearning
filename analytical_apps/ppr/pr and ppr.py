import networkx as nx

# 创建一个有向图
G = nx.DiGraph()
G.add_edges_from([(2, 1), (2, 3)])

# 计算普通Pagerank
pagerank_scores = nx.pagerank(G)

# 打印每个节点的Pagerank得分
for node, score in pagerank_scores.items():
    print("Node:", node, "Pagerank Score:", score)

# 计算以节点6为单源的SSPPR
ssppr_scores = nx.pagerank(G, personalization={2: 1})

# 打印每个节点的SSPPR得分
for node, score in ssppr_scores.items():
    print("Node:", node, "SSPPR Score:", score)