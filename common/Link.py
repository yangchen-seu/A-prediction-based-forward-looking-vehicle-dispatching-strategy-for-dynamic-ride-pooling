class Link:
    def __init__(self, id, from_node_id, to_node_id, distance) -> None:
        self.id = id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.distance = distance


    def setLinks(self, links):
        self.links = links