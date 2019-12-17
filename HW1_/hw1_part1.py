import random
import networkx as nx
import math
from collections import Counter
import numpy as np
from numpy import inf
# Necessary in order to run on the server, without display
from matplotlib import use

use('Agg')

from matplotlib import pyplot as plt



'''
Question 2 parts 2.a-2.c
'''

def H_features(time):

    MG = nx.MultiDiGraph()
    H = nx.Graph()
    with open('graph_by_time.txt', 'r') as data:
        next(data)
        for line in data:
            u, v, timestemp = line.split()
            u = int(u)
            v = int(v)
            timestemp = int(timestemp)
            MG.add_nodes_from([u, v])
            MG.add_edge(u, v)
            H.add_nodes_from([u, v])
            if timestemp <= time:
                H.add_edge(u, v, weight='w')
                if MG.has_edge(v, u):
                    H.edges[u, v]['weight'] = 's'

    sp = dict(nx.all_pairs_shortest_path_length(H))


    clustering_coefficient={}
    closeness_centrality={}
    for node in H.nodes:
        if H.degree(node) < 2:
            cls  = 0

        openTriangles = 0
        triangles = 0

        neighbors = list(H.neighbors(node))
        for b in neighbors:
            for c in neighbors:
                if b is not c and H.has_edge(b, c) and b < c:
                    triangles += 1
                if b is not c and not H.has_edge(b, c) and b < c:
                    openTriangles += 1

        if openTriangles == 0 and triangles == 0:
            cls=0
        else:
            cls = ((triangles) / (openTriangles + (triangles)))
        clustering_coefficient[node]=(cls)

        num = nx.node_connected_component(H, node)
        clo = 0.0
        totsp = sum(sp[node].values())
        if totsp > 0.0 and len(H) > 1:
            clo = (len(sp[node]) - 1.0) / totsp
            s = (len(sp[node]) - 1.0) / (len(H) - 1)
            clo *= s
        else:
            clo = 0.0

        closeness_centrality[node]= clo


    def bfs(G, s):  # bfs for all shortes path
        S = []
        P = {}
        for v in G:
            P[v] = []
        sigma = dict.fromkeys(G, 0.0)
        D = {}
        sigma[s] = 1.0
        D[s] = 0
        Q = [s]
        while Q:  # while Q is not empty
            vi = Q.pop(0)
            S.append(vi)
            Dv = D[vi]
            sigmav = sigma[vi]
            for w in G[vi]:
                if w not in D:
                    Q.append(w)
                    D[w] = Dv + 1
                if D[w] == Dv + 1:  # relax
                    sigma[w] += sigmav
                    P[w].append(vi)  # predecessors
        return S, P, sigma


    def calc_flow_val(betweenness, S, P, sigma, s):
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            flow_val = (1.0 + delta[w]) / sigma[w]
            for vv in P[w]:
                delta[vv] += sigma[vv] * flow_val
            if w != s:
                betweenness[w] += delta[w]

        return betweenness


    def betweenness_centrality(G):
        betweenness = dict.fromkeys(G, 0.0)
        for s in G.nodes:
            S, P, sigma = bfs(G, s)
            betweenness = calc_flow_val(betweenness, S, P, sigma, s)

        n = G.number_of_nodes()
        nCr = math.factorial(n - 1) / (math.factorial(2) * math.factorial(n - 3))
        multiplier = (1 / nCr)

        for bet in betweenness:
            betweenness[bet] = betweenness[bet] / 2 * multiplier

        return betweenness

    node=0
    nodes_feat_dict = {}
    betweenness = betweenness_centrality(H)

    degreeHist = {}

    for node in H:
        nodes_feat_dict[node]=[H.degree(node), clustering_coefficient[node], closeness_centrality[node],betweenness[node]]
    return nodes_feat_dict





"""
Question 3
"""


def calc_no(time):
    MG = nx.MultiDiGraph()
    H = nx.Graph()
    with open('graph_by_time.txt', 'r') as data:
        next(data)
        for line in data:
            u, v, timestemp = line.split()
            u = int(u)
            v = int(v)
            timestemp = int(timestemp)
            MG.add_nodes_from([u, v])
            MG.add_edge(u, v)
            H.add_nodes_from([u, v])
            if timestemp <= time:
                H.add_edge(u, v, weight='w')
                if MG.has_edge(v, u):
                    H.edges[u, v]['weight'] = 's'

    overlap = {}
    for edge in H.edges():
        u, v = edge
        uN = set(H.neighbors(u))
        vN = set(H.neighbors(v))
        uvIntersection = uN.intersection(vN)
        uvUnion = uN.union(vN)
        if uvIntersection is not 0 and len(uvUnion) > 2:
            overlap[edge] = len(uvIntersection) / (len(uvUnion)-2)
        else:
            overlap[edge] = 0
    return overlap

'''
Question 4
'''


def stc_index(time):
    MG = nx.MultiDiGraph()
    H = nx.Graph()
    with open('graph_by_time.txt', 'r') as data:
        next(data)
        for line in data:
            u, v, timestemp = line.split()
            u = int(u)
            v = int(v)
            timestemp = int(timestemp)
            MG.add_nodes_from([u, v])
            MG.add_edge(u, v)
            H.add_nodes_from([u, v])
            if timestemp <= time:
                H.add_edge(u, v, weight='w')
                if MG.has_edge(v, u):
                    H.edges[u, v]['weight'] = 's'
    betStrong = 0
    allStrong = 0
    stc_index = {}
    for nodee in H.nodes:
        if H.degree(nodee) < 2:
            stc = 0

        neighbors = list(H.neighbors(nodee))
        for a in neighbors:
            if H.edges[nodee, a]['weight'] == 's':
                allStrong += 1

        for b in neighbors:
            for c in neighbors:
                if H.edges[nodee, b]['weight'] == 's' and H.edges[nodee, c]['weight'] == 's' and H.has_edge(b, c) and b is not c and b < c:
                    betStrong += 1

        allStrong_2 = allStrong - 2
        if allStrong_2 <= 0:
            stc=0
        else:
            multipler = math.factorial(allStrong) / (math.factorial(allStrong_2) * math.factorial(2))

            stc=((betStrong) / (multipler))

        stc_index[nodee] = stc

    return stc_index
