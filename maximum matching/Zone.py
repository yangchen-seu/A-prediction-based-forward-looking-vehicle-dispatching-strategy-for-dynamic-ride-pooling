class Zone:
    def __init__(self, id, left_x, left_y, right_x,right_y) -> None:
        self.id = id
        self.left_x = left_x
        self.left_y = left_y
        self.right_x = right_x
        self.right_y = right_y
        self.nodes = []

    def show(self):
        print('self.id{},self.left_x{},self.left_y{} ,self.right_x{},self.right_y{}'.format(self.id,self.left_x
        ,self.left_y 
        ,self.right_x
        ,self.right_y))
        print('nodes:',self.nodes)