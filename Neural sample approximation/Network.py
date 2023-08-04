
from collections import defaultdict
from heapq import *

import pandas as pd
import Node


# 网格城市
class Network:
    
    # initial 
    def __init__(self) -> None:
        self.node = pd.read_csv('input\\node.csv')
        self.link = pd.read_csv('input\\links.csv')

        self.Nodes = []
        self.edges = []

        ### --- Read the topology, and generate all edges in the given topology.
        self.loadNodes()
        self.loadLinks()
    

    # 建立网格
    def loadNodes(self):
        for index,row in self.node.iterrows():
            self.Nodes.append(Node.Node(row['node_id'], lat = row['y_coord'],lon = row['x_coord'],))
 
    def loadLinks(self):
        for index,row in self.link.iterrows():
            self.edges.append((row['from_node_id'],row['to_node_id'],row['length']))


    def dijkstra_raw(self, edges, from_node, to_node):
        g = defaultdict(list)
        for l,r,c in edges:
            g[l].append((c,r))
        q, seen = [(0,from_node,())], set()
        while q:
            (cost,v1,path) = heappop(q)
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)
                if v1 == to_node:
                    return cost,path
                for c, v2 in g.get(v1, ()):
                    if v2 not in seen:
                        heappush(q, (cost+c, v2, path))
        return float("inf"),[]


    def dijkstra(self, edges, from_node, to_node):
        len_shortest_path = -1
        ret_path=[]
        length,path_queue = self.dijkstra_raw(edges, from_node, to_node)
        if len(path_queue)>0:
            len_shortest_path = length		## 1. Get the length firstly;
            ## 2. Decompose the path_queue, to get the passing nodes in the shortest path.
            left = path_queue[0]
            ret_path.append(left)		## 2.1 Record the destination node firstly;
            right = path_queue[1]
            while len(right)>0:
                left = right[0]
                ret_path.append(left)	## 2.2 Record other nodes, till the source-node.
                right = right[1]
            ret_path.reverse()	## 3. Reverse the list finally, to make it be normal sequence.
        return len_shortest_path,ret_path


    # 返回两点之间的最短路
    def get_path(self, O, D):
        length,Shortest_path = self.dijkstra(self.edges, O, D)
        return length


    def test(self):
        self.Nodes[0].show()
        self.Nodes[10].show()
        print(self.get_path(0,10))


# net = Network()
# net.test()
