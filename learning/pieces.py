import move

class piece:
    def __init__(self, x, y, side):
        self.moved=False
        self.side=side
        self.x=x
        self.y=y

    def describe(self):
        print(" I am a "+str(self.side)+" "+str(self.name)+" on X = "+str(self.x)+" and Y = "+str(self.y)+"\n")


class pawn(piece):
    def __init__(self, x, y, side):
        super(pawn, self).__init__(x, y, side)
        self.name="pawn"
        self.id="W-P" if side=="white" else "B-P"

    def move(self, x, y, board):
        if not self.moved and self.y==y and ((self.side=="black" and self.x-x>-3 and self.x-x<0) or (self.side=="white" and self.x-x>0 and self.x-x<3 )):
            print("Enabled movement")
            self.x=x
            self.y=y
            self.moved=True
            return True
        elif self.y==y and ((self.side=="black" and self.x-x>-2 and self.x-x<0) or (self.side=="white" and self.x-x>0 and self.x-x<2 )):
            print("Enabled movement")
            self.x=x
            self.y=y
            self.moved=True
            return True
        elif board.getpiece(x, y)!=None and board.getpiece(x, y).side!=self.side and (self.y-y==-1 or self.y-y==1) and ((self.side=="black" and self.x-x>-2 and self.x-x<0) or (self.side=="white" and self.x-x>0 and self.x-x<2)):
            print("Enabled movement (eating)")
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False

    def moves(self):
        l=[]
        if self.side=="white":
            l.append(move(self.x-1))
            if not self.moved:
                l.append(move(self.x-2))
        else:
            l.append(move(self.x+1))
            if not self.moved:
                l.append(move(self.x+2))
                
    
class knight(piece):
    def __init__(self, x, y, side):
        super(knight, self).__init__(x, y, side)
        self.name="knight"
        self.id="W-KN" if side=="white" else "B-KN"

    def move(self, x, y, board):
        if ((self.x-x==2 or self.x-x==-2) and (self.y-y==1 or self.y-y==-1)) or ((self.y-y==2 or self.y-y==-2) and (self.x-x==1 or self.x-x==-1)):
            print("Enabled movement")
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False


class bishop(piece):
    def __init__(self, x, y, side):
        super(bishop, self).__init__(x, y, side)
        self.name="bishop"
        self.id="W-B" if side=="white" else "B-B"

    def move(self, x, y, board):
        if self.x-x==self.y-y or self.x-x==(-1)*(self.y-y):
            print("Enabled movement")
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False

class rook(piece):
    def __init__(self, x, y, side):
        super(rook, self).__init__(x, y, side)
        self.name="rook"
        self.id="W-R" if side=="white" else "B-R"

    def move(self, x, y, board):
        if self.y==y or self.x==x:
            print("Enabled movement")
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False

class queen(piece):
    def __init__(self, x, y, side):
        super(queen, self).__init__(x, y, side)
        self.name="queen"
        self.id="W-Q" if side=="white" else "B-Q"

    def move(self, x, y, board):
        if (self.y==y or self.x==x) or (self.x-x==self.y-y or self.x-x==(-1)*self.y-y):
            print("Enabled movement")
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False

class king(piece):
    def __init__(self, x, y, side):
        super(king, self).__init__(x, y, side)
        self.name="king"
        self.id="W-KI" if side=="white" else "B-KI"

    def move(self, x, y, board):
        if self.x-x<2 and self.x-x>-2 and self.y-y<2 and self.y-y>-2:
            print("Enabled movement")
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False
