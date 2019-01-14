from move import *

class piece:
    def __init__(self, x, y, side):
        self.moved=False
        self.side=side
        self.x=x
        self.y=y
        self.historyx=[]
        self.historyy=[]
        self.historyx.append(x)
        self.historyy.append(y)
    def revoke(self):
        m=move(self.x, self.y, self.historyx[-1], self.historyy[-1],self.name, self.side)
        self.x=self.historyx.pop()
        self.y=self.historyy.pop()
        if len(self.historyx)==1:
            self.moved=False
        return m
        
    def addsamemove(self):
        self.historyx.append(self.x)
        self.historyy.append(self.y)

    def forcemove(self,x,y):
        self.x=x
        self.y=y
        self.historyx.append(self.x)
        self.historyy.append(self.y)
    def describe(self):
        print(" I am a "+str(self.side)+" "+str(self.name)+" on X = "+str(self.x)+" and Y = "+str(self.y)+"\n")


class pawn(piece):
    def __init__(self, x, y, side):
        super(pawn, self).__init__(x, y, side)
        self.name="pawn"
        self.id="W-P" if side=="white" else "B-P"

    def move(self, x, y, board):
        if not self.moved and self.y==y and ((self.side=="black" and self.x-x>-3 and self.x-x<0) or (self.side=="white" and self.x-x>0 and self.x-x<3 )):
        
            self.historyx.append(self.x)
            self.historyy.append(self.y)
            self.x=x
            self.y=y
            self.moved=True
            return True
        elif self.y==y and ((self.side=="black" and self.x-x>-2 and self.x-x<0) or (self.side=="white" and self.x-x>0 and self.x-x<2 )):
            
            self.historyx.append(self.x)
            self.historyy.append(self.y)
            self.x=x
            self.y=y
            
            self.moved=True
            return True
        elif board.getpiece(x, y)!=None and board.getpiece(x, y).side!=self.side and (self.y-y==-1 or self.y-y==1) and ((self.side=="black" and self.x-x>-2 and self.x-x<0) or (self.side=="white" and self.x-x>0 and self.x-x<2)):
            
            self.historyx.append(self.x)
            self.historyy.append(self.y)
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False

    def moves(self, board):
        l=[]
        if self.side=="white":
            l.append(move(self.x, self.y, self.x-1, self.y, self.name, self.side))
            if not self.moved:
                l.append(move(self.x, self.y,self.x-2, self.y, self.name, self.side))
        else:
            l.append(move(self.x, self.y, self.x+1, self.y, self.name, self.side))
            if not self.moved:
                l.append(move(self.x, self.y, self.x+2, self.y, self.name, self.side))
                
        if board.getpiece(self.x-1,self.y-1)!=None and self.side=="white" and board.getpiece(self.x-1,self.y-1).side=="black":
            l.append(move(self.x, self.y, self.x-1, self.y-1, self.name, self.side))
        if board.getpiece(self.x-1,self.y+1)!=None and self.side=="white" and board.getpiece(self.x-1,self.y+1).side=="black":
            l.append(move(self.x, self.y, self.x-1, self.y+1, self.name, self.side))
        if board.getpiece(self.x+1,self.y-1)!=None and self.side=="black" and board.getpiece(self.x+1,self.y-1).side=="white":
            l.append(move(self.x, self.y, self.x+1, self.y-1, self.name, self.side))
        if board.getpiece(self.x+1,self.y+1)!=None and self.side=="black" and board.getpiece(self.x+1,self.y+1).side=="white":
            l.append(move(self.x, self.y, self.x+1, self.y+1, self.name, self.side))    
        return l
class knight(piece):
    def __init__(self, x, y, side):
        super(knight, self).__init__(x, y, side)
        self.name="knight"
        self.id="W-KN" if side=="white" else "B-KN"

    def move(self, x, y, board):
        if ((self.x-x==2 or self.x-x==-2) and (self.y-y==1 or self.y-y==-1)) or ((self.y-y==2 or self.y-y==-2) and (self.x-x==1 or self.x-x==-1)):
            
            self.historyx.append(self.x)
            self.historyy.append(self.y)
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False
    
    def moves(self, board):
        l=[]
        l.append(move(self.x, self.y, self.x+2, self.y+1, self.name, self.side))
        l.append(move(self.x, self.y, self.x+2, self.y-1, self.name, self.side))
        l.append(move(self.x, self.y, self.x-2, self.y+1, self.name, self.side))
        l.append(move(self.x, self.y, self.x-2, self.y-1, self.name, self.side))
        l.append(move(self.x, self.y, self.x+1, self.y+2, self.name, self.side))
        l.append(move(self.x, self.y, self.x+1, self.y-2, self.name, self.side))
        l.append(move(self.x, self.y, self.x-1, self.y+2, self.name, self.side))
        l.append(move(self.x, self.y, self.x-1, self.y-2, self.name, self.side))
        return l


class bishop(piece):
    def __init__(self, x, y, side):
        super(bishop, self).__init__(x, y, side)
        self.name="bishop"
        self.id="W-B" if side=="white" else "B-B"

    def move(self, x, y, board):
        if self.x-x==self.y-y or self.x-x==(-1)*(self.y-y):
            
            self.historyx.append(self.x)
            self.historyy.append(self.y)
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False

    def moves(self, board):
        l=[]
        i=self.x-8
        j=self.y-8
        while i<self.x+8 and j<self.y+8:
            l.append(move(self.x, self.y, i, j, self.name, self.side))
            i=i+1
            j=j+1

        return l

class rook(piece):
    def __init__(self, x, y, side):
        super(rook, self).__init__(x, y, side)
        self.name="rook"
        self.id="W-R" if side=="white" else "B-R"

    def move(self, x, y, board):
        if self.y==y or self.x==x:
            
            self.historyx.append(self.x)
            self.historyy.append(self.y)
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False
    def moves(self, board):
        l=[]
        for i in range(self.x-8, self.x+8):
            l.append(move(self.x, self.y, i, self.y, self.name, self.side))
        for j in range(self.y-8, self.y+8):
            l.append(move(self.x, self.y, self.x, j, self.name, self.side))
        return l

class queen(piece):
    def __init__(self, x, y, side):
        super(queen, self).__init__(x, y, side)
        self.name="queen"
        self.id="W-Q" if side=="white" else "B-Q"

    def move(self, x, y, board):
        if (self.y==y or self.x==x) or (self.x-x==self.y-y or self.x-x==(-1)*(self.y-y)):
            
            self.historyx.append(self.x)
            self.historyy.append(self.y)
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False
    def moves(self, board):
        l=[]
        i=self.x-8
        j=self.y-8
        while i<self.x+8 and j<self.y+8:
            l.append(move(self.x, self.y, i, j, self.name, self.side))
            i=i+1
            j=j+1
        for i in range(self.x-8, self.x+8):
            l.append(move(self.x, self.y, i, self.y, self.name, self.side))
        for j in range(self.y-8, self.y+8):
            l.append(move(self.x, self.y, self.x, j, self.name, self.side))
        return l
class king(piece):
    def __init__(self, x, y, side):
        super(king, self).__init__(x, y, side)
        self.name="king"
        self.id="W-KI" if side=="white" else "B-KI"

    def move(self, x, y, board):
        if self.x-x<2 and self.x-x>-2 and self.y-y<2 and self.y-y>-2:
            
            self.historyx.append(self.x)
            self.historyy.append(self.y)
            self.x=x
            self.y=y
            self.moved=True
            return True
        else:
            print("Disabled movement")
            return False
    
    def moves(self, board):
        l=[]
        l.append(move(self.x, self.y, self.x+1, self.y, self.name, self.side))
        l.append(move(self.x, self.y, self.x-1, self.y, self.name, self.side))
        l.append(move(self.x, self.y, self.x+1, self.y+1, self.name, self.side))
        l.append(move(self.x, self.y, self.x-1, self.y+1, self.name, self.side))
        l.append(move(self.x, self.y, self.x+1, self.y-1, self.name, self.side))
        l.append(move(self.x, self.y, self.x-1, self.y-1, self.name, self.side))
        l.append(move(self.x, self.y, self.x, self.y+1, self.name, self.side))
        l.append(move(self.x, self.y, self.x, self.y-1, self.name, self.side))
        return l

