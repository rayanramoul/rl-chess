from pieces import *

class board:
    def __init__(self, board=None, move=None):
        self.board=[['O' for x in range(8)] for y in range(8)]
        self.game="on"
        self.turn="white"
        self.pieces=[]
        for i in range(8):
            self.add(pawn(6,i,"white"))
            self.add(pawn(1,i,"black"))

        self.add(knight(7,1,"white"))
        self.add(knight(7,6,"white"))
        self.add(knight(0,1,"black"))
        self.add(knight(0,6,"black"))

        self.add(rook(7,0,"white"))
        self.add(rook(7,7,"white"))
        self.add(rook(0,0,"black"))
        self.add(rook(0,7,"black"))

        self.add(bishop(7,2,"white"))
        self.add(bishop(7,5,"white"))
        self.add(bishop(0,2,"black"))
        self.add(bishop(0,5,"black"))

        self.add(king(7,3,"white"))
        self.add(queen(7,4,"white"))
        self.add(king(0,4,"black"))
        self.add(queen(0,3,"black"))
        self.precmoves=[]
        self.moves=[]
        self.allmoves()
        self.draw()
    def check(self):
        l=[]
        r=False
        for i in self.pieces:
            for j in i.moves(self):
                try:
                    if (j.baseside!=self.getpiece(j.newx, j.newy).side) and j.newx>-1 and j.newx<8 and j.newy>-1 and j.newy<8 and j.baseside!=self.turn and not(self.blocked(j.basex, j.basey, j.newx, j.newy)):
                        l.append(j)
                        if self.getpiece(j.newx, j.newy)!=None and (self.getpiece(j.newx, j.newy).side==self.turn and self.getpiece(j.newx, j.newy).name=="king"):
                            r=True
                except:
                    if j.newx>-1 and j.newx<8 and j.newy>-1 and j.newy<8 and j.baseside!=self.turn and not(self.blocked(j.basex, j.basey, j.newx, j.newy)):
                        l.append(j)
                        if self.getpiece(j.newx, j.newy)!=None and (self.getpiece(j.newx, j.newy).side==self.turn and self.getpiece(j.newx, j.newy).name=="king"):
                            r=True
        if r:
            print("CHEck !!!!!")



        



    def print(self):
        print('\n\n'.join(['\t'.join(['{:4}'.format(item) for item in row]) for row in self.board]))

    def add(self, element):
        self.pieces.append(element)
        self.draw()

    def draw(self):
        for i in self.pieces:
            self.board[i.x][i.y]=i.id

    def getpiece(self, x, y):
        i=None
        for i in self.pieces:
            if i.x==x and i.y==y:
                return i

    def removepiece(self, x, y):
        d=0
        for i in self.pieces:
            if i.x==x and i.y==y:
                del self.pieces[d]
                self.board[x][y]='O'
                return
            d=d+1

    def blocked(self, basex, basey, newx, newy):
        if self.getpiece(basex, basey).name=="knight":
            return False
        else:
            if basex==newx:
                print("OUI 1")
                if basey>newy:
                    k=-1
                else:
                    k=1
                for i in range(basey+k,newy):
                    if self.getpiece(basex,i)!=None:
                        return True
                
                for i in range(newy+k,basey):
                    if self.getpiece(basex,i)!=None:
                        return True
                
            elif basey==newy:
                print("OUI 2 basex="+str(basex)+" newx="+str(newx))
                if basex>newx:
                    k=-1
                else:
                    k=1
                for i in range(basex+k,newx):
                    print("i : "+str(i))
                    if self.getpiece(i,basey)!=None:
                        return True
                             
                for i in range(newx+k,basex):
                    print("i : "+str(i))
                    if self.getpiece(i,basey)!=None:
                        return True
                
            else:
                print("OUI 3")
                if newx-basex>0:
                    incx=1
                else:
                    incx=-1
                if newy-basey>0:
                    incy=1
                else:
                    incy=-1

                j=basey+incy
                i=basex+incx
                while j!=newy and i!=newx:
                    if self.getpiece(i,j)!=None:
                        return True
                    i=i+incx
                    j=j+incy

            return False

    def allmoves(self):
        l=[]
        self.precmoves=self.moves
        for i in self.pieces:
            for j in i.moves(self):
                try:
                    if (j.baseside!=self.getpiece(j.newx, j.newy).side) and j.newx>-1 and j.newx<8 and j.newy>-1 and j.newy<8 and j.baseside==self.turn and not(self.blocked(j.basex, j.basey, j.newx, j.newy)):
                        l.append(j)
 #                       j.describe()
                except:
                    if j.newx>-1 and j.newx<8 and j.newy>-1 and j.newy<8 and j.baseside==self.turn and not(self.blocked(j.basex, j.basey, j.newx, j.newy)):
                        l.append(j)
#                        j.describe()
        self.moves=l
        return l
                    
    def move(self,basex, basey, newx, newy):
        if self.getpiece(basex, basey)==None:
            print("You're trying to move nothing !")
            return False
        if self.getpiece(basex, basey).side!=self.turn:
            print("Move your own pieces !")
            return False
        print("BLOCKED MOVE ??????")
        if self.blocked(basex, basey, newx, newy):
            print("There is another piece on the road ! ")
            return False
        if basex>7 or basex<0 or basey>7 or basex<0:
            print("Out of Board movement !")
            return False
        elif self.board[newx][newy]!='O' and self.getpiece(basex, basey).side==self.getpiece(newx, newy).side:
            print("You can't eat from your own side !")
            return False
        else:
            for i in self.pieces:
                if i.x==basex and i.y==basey:
                    if i.move(newx, newy, self):
                        if self.turn=="white":
                            self.turn="black"
                        else:
                            self.turn="white"
                        self.board[basex][basey]='O'
                        d=0
                        for o in self.pieces:
                            if o.x==newx and o.y==newy and i is not o:
                                del self.pieces[d]
                            d=d+1
                        self.draw()
        print("\n\n")
        self.allmoves()
#        self.print()
        self.check()
        return True
