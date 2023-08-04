'''
Author: your name
Date: 2021-12-16 21:35:28
LastEditTime: 2021-12-16 21:41:39
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\Reinforcementlearning\Location.py
'''
class Node:
    def __init__(self, id, lat, lon) -> None:
        self.id = id
        self.lat = lat
        self.lon = lon
        self.zone = 0

    def setLinks(self, links):
        self.links = links
    
    def show(self):
        print('id:{}, lat:{}, lon:{}'.format(self.id, self.lat, self.lon))

    def setZone(self, zone):
        self.zone = zone
    
    def getZone(self):
        return self.zone