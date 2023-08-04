
from collections import defaultdict
from heapq import *

import pandas as pd
import numpy as np
import Node
import Zone
from numba import jit

# 网格城市
class Network:
    
    # initial 
    def __init__(self) -> None:
        self.node = pd.read_csv('input\\node.csv')
        self.link = pd.read_csv('input\\links.csv')

        self.Nodes = {}
        self.edges = []
        self.zones = {}

        ### --- Read the topology, and generate all edges in the given topology.
        min_x,min_y,max_x,max_y = self.loadNodes()
        self.loadZones( min_x,min_y,max_x,max_y )
        self.loadLinks()
        # 给节点建立网格索引
        for key in self.Nodes.keys():
            self.generateIndex(key)
            # print('node id:{},zone id:{}'.format(self.Nodes[key].id,self.Nodes[key].getZone()))
        self.shortest_path = {}
        tmp = 0
        for key in self.zones.keys():
            tmp += len(self.zones[key].nodes)
        # print('nodes',len(self.Nodes))
        # print('zones',len(self.zones))
        # print(tmp)
        self.edges_df = pd.DataFrame(self.edges, columns= ['from_node_id','to_node_id','length'])


    # 建立网格
    def loadZones(self,min_x,min_y,max_x,max_y):
        id = 0
        r = 0.05
        for x in np.arange(min_x, max_x, r):
            for y in np.arange(min_y, max_y, r):
                
                zone = Zone.Zone(id, x,y, x+r,y+r)
                self.zones[id] = zone
                id+=1
                # print(x,y)
        

    def loadNodes(self):
        min_x = 8888
        min_y = 8888
        max_x = 0
        max_y = 0
        for index,row in self.node.iterrows():

            if  row['y_coord'] > max_y:
                max_y =  row['y_coord']
            elif row['y_coord'] < min_y:
                min_y = row['y_coord']
            
            if  row['x_coord'] > max_x:
                max_x =  row['x_coord']
            elif row['x_coord'] < min_x:
                min_x = row['x_coord']

            node = Node.Node(row['node_id'], lat = row['y_coord'],lon = row['x_coord'],)
            self.Nodes[row['node_id']] = node
        return min_x,min_y,max_x,max_y
 
    def loadLinks(self):
        for index,row in self.link.iterrows():
            self.edges.append((row['from_node_id'],row['to_node_id'],row['length']))



    # 给节点建立网格索引
    def generateIndex(self,node_key):
        node = self.Nodes[node_key]
        # print('node.lon ',node.lon ,'node.lat ',node.lat )
        for key in self.zones.keys():
            # print(',self.zones[key].left_x',self.zones[key].left_x,'self.zones[key].left_y',self.zones[key].left_y,\
            #     'self.zones[key].right_x',self.zones[key].right_x,'self.zones[key].right_y',self.zones[key].right_x)
            if node.lon > self.zones[key].left_x and node.lon < self.zones[key].right_x:
                if node.lat > self.zones[key].left_y and node.lat < self.zones[key].right_y:
                    node.setZone(self.zones[key])
                    self.zones[key].nodes.append(node.id)
    

    # 最短路径内容
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

    # 返回最短路径间每两个点的距离
    def find_length(self,Shortest_path):
        shortest_path_length = []
        for i in range(len(Shortest_path)-1):
            shortest_path_length.append(self.edges_df['length'].loc[(self.edges_df['from_node_id'] == Shortest_path[i]) & \
                (self.edges_df['to_node_id'] == Shortest_path[i+1])].values[0])
        return shortest_path_length

    # 返回两点之间的最短路
    def get_path(self, O, D):
        length,Shortest_path = self.dijkstra(self.edges, O, D)
        return length, Shortest_path[1:] , self.find_length(Shortest_path)


    def test(self):
        order_list = pd.read_csv(
            './input/order.csv')

        O_locations = order_list['O_location'].unique()
        D_locations = order_list['D_location'].unique()
        length, Shortest_path, shortest_path_length = self.get_path(O_locations[0], D_locations[0])
        print(length)
        print(len(shortest_path_length))
        print(sum(shortest_path_length))
        print(len(Shortest_path))
        print(shortest_path_length)
        print(Shortest_path)


    # @ jit()
    def save_shortest_path(self,OD):
        

        O_locations = OD['origin_id'].unique()
        D_locations = OD['destination_id'].unique()

        locations =list(set(list(O_locations) + list(D_locations)))
        print(len(O_locations),len(D_locations),len(locations))
        O = []
        D = []
        distance = []
        for i in tqdm(range(len(locations))):
            for j in range(len(locations)):
                O.append(locations[i])
                D.append(locations[j])
                distance.append(self.get_path(locations[i], locations[j])[0])
        res = pd.DataFrame({'O':O, 'D':D, 'distance':distance})
        res.to_csv('./input/shortest_path.csv')

# net = Network()
# net.test()
# from tqdm import tqdm
# OD = pd.read_csv('./input/month_5_OD.csv')
# net.save_shortest_path(OD)
