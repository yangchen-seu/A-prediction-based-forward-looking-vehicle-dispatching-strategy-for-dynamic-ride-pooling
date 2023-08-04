class Trip:
    def __init__(self, r1, r2,v) -> None:
        self.r1 = r1
        self.r2 = r2
        self.v = v
        self.value = 0
        self.reposition_target = 0
        if r1 == 'default':
            self.reposition_target = 1
        if r2 == False:
            self.target = 0 # 'pick up p1'
        else:
            self.target = 1 # 'pick up p1 and p2'

    def set_Value(self,value):
        self.value = value
    