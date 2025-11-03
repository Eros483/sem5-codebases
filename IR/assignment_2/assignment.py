"""
Title: Implementation of Page Rank and HITS Algorithm on a Small Directed Social Network for Information Retrieval Assignment
Author: Arnab Mandal

Description:
    This code implements the Page Rank and HITS algorithms on a small directed social network graph.
    It calculates the Page Rank scores, Authority scores, and Hub scores for each node in the graph.
    It also visualizes the network and the scores using matplotlib and networkx.

Instructions to Run:
    1. Ensure Python is installed.
    2. run `pip install numpy networkx matplotlib pandas` to install required libraries.
    3. Save this script as `assignment.py`.
    4. Run the script using `python assignment.py`.
    5. Visualizations will be saved as `pagerank_hits_analysis.png`.
    6. Analysis of the results is provided in `analysis.txt`.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd

# ============================================================================>
# Small directed network for a social network (users retweeting each other)
# ============================================================================>
edges=[
    ('Alice', 'Bob'), ('Alice', 'Charlie'), ('Alice', 'Diana'),
    ('Bob', 'Alice'), ('Bob', 'Eve'), ('Bob', 'Frank'),
    ('Charlie', 'Alice'), ('Charlie', 'Diana'), ('Charlie', 'George'),
    ('Diana', 'Alice'), ('Diana', 'Eve'), ('Diana', 'Hannah'),
    ('Eve', 'Bob'), ('Eve', 'Frank'), ('Eve', 'Ian'),
    ('Frank', 'Bob'), ('Frank', 'George'), ('Frank', 'Jack'),
    ('George', 'Charlie'), ('George', 'Hannah'), ('George', 'Diana'),
    ('Hannah', 'Diana'), ('Hannah', 'Ian'), ('Hannah', 'Alice'),
    ('Ian', 'Eve'), ('Ian', 'Jack'), ('Ian', 'Hannah'),
    ('Jack', 'Frank'), ('Jack', 'Ian'), ('Jack', 'Bob')
]

Graph=nx.DiGraph()
Graph.add_edges_from(edges)

def pageRank(G, epsilon=0.15, convergence=0.0001, max_iter=100):
    """
    PageRank Algorithm Implementation with E=0.15 with iterations until convergence  
    Args:
        G: Social Network Graph
        epsilon: Teleportation Probability as instructed in assignment
        convergence: Convergence threshold of 0.0001 as instructed in assignment
        max_iter: Maximum iterations for safeguarding against infinite runs
    
    Returns:
        dict: Page Rank scores for each node as per algorithm
    """
    nodes=list(G.nodes())
    n=len(nodes)
    
    pr={
        node: 1.0/n for node in nodes
    }
    
    iterations=0
    converged=False
    
    while not converged and iterations<max_iter:
        new_pr={}
        
        for node in nodes:
            rank_sum=0
            for in_neighbor in G.predecessors(node):
                out_degree=G.out_degree(in_neighbor)
                if out_degree>0:
                    rank_sum+=pr[in_neighbor]/out_degree
            
            new_pr[node]=epsilon/n+(1-epsilon)*rank_sum

        max_diff= max(abs(new_pr[node] - pr[node]) for node in nodes)
        converged= max_diff<convergence
        
        pr=new_pr
        iterations+=1
    
    return pr

pr_scores=pageRank(Graph)

print("="*100)
print("Top 5 Ranked nodes as per Page Rank Algorithm with Parameters as specified in Assignment")
print("="*100)

top_pr = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
for i, (node, score) in enumerate(top_pr, 1):
    print(f"  {i}. {node:12s}: {score:.6f}")
print()

def hits(G, max_iter=100, convergence=0.0001):
    """
    Implementation of HITS Algorithm with iterations until convergence.
    
    Args:
        G: Social Network Graph
        max_iter: Maximum iterations for safeguarding against infinite runs
        convergence: Convergence threshold of 0.0001 as instructed in assignment
    
    Returns:
        dict: Dictionary of Authority scores
        dict: Dictionary of Hub scores
    """
    nodes=list(G.nodes())

    auth={
        node: 1.0 for node in nodes
    }
    hub={
        node: 1.0 for node in nodes
    }
    
    iterations=0
    converged=False
    
    while not converged and iterations<max_iter:
        new_auth={}
        for node in nodes:
            new_auth[node]=sum(hub[pred] for pred in G.predecessors(node))
        
        new_hub={}
        for node in nodes:
            new_hub[node]=sum(new_auth[succ] for succ in G.successors(node))
        
        auth_norm=np.sqrt(sum(score**2 for score in new_auth.values()))
        if auth_norm>0:
            new_auth={node: score/auth_norm for node, score in new_auth.items()}

        hub_norm=np.sqrt(sum(score**2 for score in new_hub.values()))
        if hub_norm>0:
            new_hub={node: score/hub_norm for node, score in new_hub.items()}

        max_diff=max(
            max(abs(new_auth[node]-auth[node]) for node in nodes),
            max(abs(new_hub[node]-hub[node]) for node in nodes)
        )
        converged=max_diff<convergence
        
        auth=new_auth
        hub=new_hub
        iterations+=1
    
    return auth, hub

auth_scores,hub_scores= hits(Graph)

print("=" * 100)
print("Top 5 Authority and Hub nodes as per HITS Algorithm with Parameters as specified in Assignment")
print("=" * 100)

top_auth=sorted(auth_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 Authority Nodes->")
for i, (node, score) in enumerate(top_auth, 1):
    print(f"  {i}. {node:12s}: {score:.6f}")
print()

top_hub=sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 Hub Nodes->")
for i, (node, score) in enumerate(top_hub, 1):
    print(f"  {i}. {node:12s}: {score:.6f}")
print()


pr_top_nodes = set(node for node, _ in top_pr)
auth_top_nodes = set(node for node, _ in top_auth)
common_nodes = pr_top_nodes.intersection(auth_top_nodes)

# ============================================================================>
# Visualization of The social network and Scores as per Page Rank algorithm and HITS
# ============================================================================>
fig, axes=plt.subplots(2, 2, figsize=(16, 14))
max_pr=max(pr_scores.values())
node_sizes=[3000*(pr_scores[node]/max_pr) for node in Graph.nodes()]

top_3_auth=set(node for node, _ in top_auth[:3])
top_3_hub=set(node for node, _ in top_hub[:3])

node_colors = []
for node in Graph.nodes():
    if node in top_3_auth and node in top_3_hub:
        node_colors.append('#9333ea')  # Purple-High Authority and Hub Scores
    elif node in top_3_auth:
        node_colors.append('#ef4444')  # Red-High Authority Scores only
    elif node in top_3_hub:
        node_colors.append('#3b82f6')  # Blue-High Hub scores only
    else:
        node_colors.append('#64748b')  # Gray-Neither authorities or hubs

pos=nx.spring_layout(Graph, k=2, iterations=50, seed=42)

# ============================================================================>
# Network Graph Plot
# ============================================================================>
ax1 = axes[0, 0]
nx.draw_networkx_nodes(Graph, pos, node_size=node_sizes, node_color=node_colors, 
                       ax=ax1, alpha=0.9, edgecolors='white', linewidths=2)
nx.draw_networkx_labels(Graph, pos, font_size=9, font_weight='bold', ax=ax1)
nx.draw_networkx_edges(Graph, pos, edge_color='#cbd5e1', arrows=True, 
                       arrowsize=15, ax=ax1, alpha=0.6, width=1.5,
                       connectionstyle='arc3,rad=0.1')
ax1.set_title('Network Graph', fontsize=14, fontweight='bold')
ax1.axis('off')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#ef4444', label='Top Authority Only'),
    Patch(facecolor='#3b82f6', label='Top Hub Only'),
    Patch(facecolor='#9333ea', label='Both Authority and Hub'),
    Patch(facecolor='#64748b', label='Neither Authority nor Hub')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

# ============================================================================>
# Page Rank Scores plot
# ============================================================================>
ax2 = axes[0, 1]
pr_sorted = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
nodes_pr = [node for node, _ in pr_sorted]
values_pr = [score for _, score in pr_sorted]
bars2 = ax2.barh(nodes_pr, values_pr, color='#8b5cf6')
ax2.set_title('PageRank Scores', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
for i, bar in enumerate(bars2):
    ax2.text(bar.get_width(), 
            bar.get_y() + bar.get_height()/2, 
            f' {values_pr[i]:.4f}', 
            va='center', 
            fontsize=7
    )

# ============================================================================>
# Authority Scores plot
# ============================================================================>
ax3 =axes[1, 0]
auth_sorted = sorted(auth_scores.items(), key=lambda x: x[1], reverse=True)
nodes_auth = [node for node, _ in auth_sorted]
values_auth = [score for _, score in auth_sorted]
bars3 = ax3.barh(nodes_auth, values_auth, color='#ef4444')
ax3.set_title('Authority Scores for all Nodes', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
for i, bar in enumerate(bars3):
    ax3.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
             f' {values_auth[i]:.4f}', va='center', fontsize=9)

# ============================================================================>
# Hub Scores plot
# ============================================================================>
ax4 = axes[1, 1]
hub_sorted = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)
nodes_hub = [node for node, _ in hub_sorted]
values_hub = [score for _, score in hub_sorted]
bars4 = ax4.barh(nodes_hub, values_hub, color='#3b82f6')
ax4.set_title('Hub Scores for all Nodes', fontsize=14, fontweight='bold')
ax4.invert_yaxis()
for i, bar in enumerate(bars4):
    ax4.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
             f' {values_hub[i]:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('pagerank_hits_analysis.png', dpi=300, bbox_inches='tight')

print("="*100)
print("Visualization saved as 'pagerank_hits_analysis.png'")
print("="*100)