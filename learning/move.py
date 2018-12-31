class move:
    def __init__(self, basex, basey, newx, newy, basepiece, baseside):
        self.basex=basex
        self.basey=basey
        self.newx=newx
        self.newy=newy
        self.basepiece=basepiece
        self.baseside=baseside
    
    def describe(self):
        print("["+str(self.baseside)+","+str(self.basepiece)+","+str(self.basex)+","+str(self.basey)+","+str(self.newx)+","+str(self.newy)+"]")

    def string(self):
        return "["+str(self.baseside)+","+str(self.basepiece)+","+str(self.basex)+","+str(self.basey)+","+str(self.newx)+","+str(self.newy)+"]"
