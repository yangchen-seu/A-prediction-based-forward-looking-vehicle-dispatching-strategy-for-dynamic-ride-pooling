class singleTrip:
    def __init__(self, r, v) -> None:
        self.r = r
        self.v = v
        self.value = 0
        self.target = 0


    def set_Value(self,value):
        self.value = value
    
    def show(self):
        print('self.r1',self.r1)
        print('self.v',self.v)
        print('self.value',self.value)
        print('self.target',self.target)


class doubleTrip:
    def __init__(self, r1, r2,v) -> None:
        self.r1 = r1
        self.r2 = r2
        self.v = v
        self.value = 0
        self.target = 1

    def set_Value(self,value):
        self.value = value
    
    def show(self):
        print('self.r1',self.r1)
        print('self.r2',self.r2)
        print('self.v',self.v)
        print('self.value',self.value)
        print('self.target',self.target)