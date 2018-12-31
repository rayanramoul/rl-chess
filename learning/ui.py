import sys
from board import board
from functools import partial
from PyQt5.QtWidgets import (QWidget, QToolTip, 
    QPushButton, QApplication, QLabel, QListWidget, QListWidgetItem)
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import QSize


class Example(QWidget):
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        super().__init__()
        self.board=board()
        self.initUI()
        
        
    def initUI(self):

        QToolTip.setFont(QFont('SansSerif', 30))
        self.clickedx=-1
        self.clickedy=-1
        
        self.buttons=[[QPushButton('Button', self) for x in range(8)] for y in range(8)]
        a=1
        b=1
        indx=0
        indy=0
        for i in self.buttons:
            for j in i:
                try:
                    k="resources/icons/"+self.board.getpiece(indx,indy).id+".png"
                    pixmap = QPixmap(k)
                    j.setText("")
                    j.setIcon(QIcon(pixmap))
                    j.setIconSize(QSize(100,100))
                    if (indy%2==0 and indx%2==0) or (indy%2==1 and indx%2==1):
                        j.setStyleSheet("background-color: #bcaaa4")
#                    j.setText(str(self.board.getpiece(indx,indy).side+" "+self.board.getpiece(indx,indy).name))
#                   j.setText(":DDD")
                except (AttributeError):
                    j.setText("")
                    if (indy%2==0 and indx%2==0) or (indy%2==1 and indx%2==1):
                        j.setStyleSheet("background-color: #bcaaa4")

                j.clicked.connect(partial(self.handleClickCase,indx,indy))
                j.resize(100,100)
                j.move(b, a)
                a=a+100
                indx=indx+1
            a=1
            indx=0
            b=b+100
            indy=indy+1       
        self.turn=QLabel(self)
        self.turn.move(850,10)

        self.possible=QLabel(self)
        self.possible.move(850,50)
        self.possible.setText("Possible Moves")

        self.possiblemoves=QListWidget(self)
        self.possiblemoves.move(830,100)
        self.possiblemoves.resize(180, 500)
        self.lastbasex=0
        self.lastbasey=0
        self.lastnewx=0
        self.lastnewy=0
        self.possiblemoves.clear()
        self.possmov=self.board.moves
        self.possiblemoves.itemClicked.connect(self.highlight)
        self.possiblemoves.itemSelectionChanged.connect(self.highlight)
        for i in self.possmov:
            item = QListWidgetItem(i.string())
            self.possiblemoves.addItem(item)
        self.turn.setFont(QFont('SansSerif', 25))
        self.turn.setText(self.board.turn+"'s turn")
        self.setGeometry(200, 300, 1024, 800)
        self.setWindowTitle('openChess')    
        self.show()

    def highlight(self, item=None):
        ind=self.possmov[self.possiblemoves.currentRow()]
        basex=ind.basex
        basey=ind.basey
        newx=ind.newx
        newy=ind.newy
        if (self.lastbasey%2==0 and self.lastbasex%2==0) or (self.lastbasey%2==1 and self.lastbasex%2==1):
            self.buttons[self.lastbasey][self.lastbasex].setStyleSheet("background-color: #bcaaa4")
        else:
            self.buttons[self.lastbasey][self.lastbasex].setStyleSheet("")
        
        if (self.lastnewy%2==0 and self.lastnewx%2==0) or (self.lastnewy%2==1 and self.lastnewx%2==1):
            self.buttons[self.lastnewy][self.lastnewx].setStyleSheet("background-color: #bcaaa4")
        else:
            self.buttons[self.lastnewy][self.lastnewx].setStyleSheet("")
        self.lastbasex=basex
        self.lastbasey=basey
        self.lastnewx=newx
        self.lastnewy=newy
        self.buttons[basey][basex].setStyleSheet("background-color: #3f51b5")
        self.buttons[newy][newx].setStyleSheet("background-color: #ff8a80")
        
    def redraw(self,beginx,beginy,endx,endy):
        if (self.lastbasey%2==0 and self.lastbasex%2==0) or (self.lastbasey%2==1 and self.lastbasex%2==1):
            self.buttons[self.lastbasey][self.lastbasex].setStyleSheet("background-color: #bcaaa4")
        else:
            self.buttons[self.lastbasey][self.lastbasex].setStyleSheet("")
        
        if (self.lastnewy%2==0 and self.lastnewx%2==0) or (self.lastnewy%2==1 and self.lastnewx%2==1):
            self.buttons[self.lastnewy][self.lastnewx].setStyleSheet("background-color: #bcaaa4")
        else:
            self.buttons[self.lastnewy][self.lastnewx].setStyleSheet("")
        self.lastbasex=-1
        self.lastbasey=-1
        self.lastnewx=-1
        self.lastnewy=-1

        self.possiblemoves.clear()
        self.possmov=self.board.moves
        for i in self.possmov:
            item = QListWidgetItem(i.string())
            self.possiblemoves.addItem(item)
            self.possiblemoves.repaint()
        self.possiblemoves.update()

        if self.board.getpiece(beginx,beginy)!=None:
            print(1)
            k="resources/icons/"+self.board.getpiece(beginx,beginy).id+".png"
            pixmap = QPixmap(k)
            self.buttons[beginy][beginx].setText("")
            self.buttons[beginy][beginx].setIcon(QIcon(pixmap))
            self.buttons[beginy][beginx].setIconSize(QSize(100,100))
            self.buttons[beginy][beginx].setText("")
            self.buttons[beginy][beginx].setCheckable(False)
            if (beginy%2==0 and beginx%2==0) or (beginy%2==1 and beginx%2==1):
                self.buttons[beginy][beginx].setStyleSheet("background-color: #bcaaa4")
            else:
                self.buttons[beginy][beginx].setStyleSheet("")
            self.buttons[beginy][beginx].update()
        else:
            self.buttons[beginy][beginx].setText("")
            self.buttons[beginy][beginx].setIcon(QIcon())
            self.buttons[beginy][beginx].setIconSize(QSize(100,100))
            self.buttons[beginy][beginx].setText("")
            self.buttons[beginy][beginx].setCheckable(False)
            if (beginy%2==0 and beginx%2==0) or (beginy%2==1 and beginx%2==1):
                self.buttons[beginy][beginx].setStyleSheet("background-color: #bcaaa4")
            else:
                self.buttons[beginy][beginx].setStyleSheet("")
            
            self.buttons[beginy][beginx].update()
        if self.board.getpiece(endx,endy)!=None:
            k="resources/icons/"+self.board.getpiece(endx,endy).id+".png"
            pixmap = QPixmap(k)
            self.buttons[endy][endx].setText("")
            self.buttons[endy][endx].setIcon(QIcon(pixmap))
            self.buttons[endy][endx].setIconSize(QSize(100,100))
            self.buttons[endy][endx].setText("")
            self.buttons[endy][endx].setCheckable(False)
            if (endy%2==0 and endx%2==0) or (endy%2==1 and endx%2==1):
                self.buttons[endy][endx].setStyleSheet("background-color: #bcaaa4")
            else:
                self.buttons[endy][endx].setStyleSheet("")

            self.buttons[endy][endx].update()
        else:
            self.buttons[endy][endx].setText("")
            self.buttons[endy][endx].setIcon(QIcon())
            self.buttons[endy][endx].setIconSize(QSize(100,100))
            self.buttons[endy][endx].setText("")
            self.buttons[endy][endx].setCheckable(False)
            if (endy%2==0 and endx%2==0) or (endy%2==1 and endx%2==1):
                self.buttons[endy][endx].setStyleSheet("background-color: #bcaaa4")
            else:
                self.buttons[endy][endx].setStyleSheet("")

            self.buttons[endy][endx].update()
        self.turn.setText(self.board.turn+"'s turn")
        self.update()
    def handleClickCase(self,indx,indy):
        print("clickedx : "+str(self.clickedx))
        print("clickedy : "+str(self.clickedy))
        print("indx : "+str(indx))
        print("indy : "+str(indy))
        self.board.allmoves()
        if self.clickedx==-1 and self.clickedy==-1:
            self.clickedx=indx
            self.clickedy=indy
            self.buttons[self.clickedy][self.clickedx].setStyleSheet("background-color: #00bfa5")
            self.buttons[self.clickedy][self.clickedx].setCheckable(True)
            self.buttons[self.clickedy][self.clickedx].style().unpolish(self.buttons[self.clickedy][self.clickedx])
            self.buttons[self.clickedy][self.clickedx].style().polish(self.buttons[self.clickedy][self.clickedx])
            self.buttons[self.clickedy][self.clickedx].update()

        elif self.board.move(self.clickedx,self.clickedy,indx,indy):
            self.redraw(self.clickedx, self.clickedy, indx, indy)
            self.clickedx=-1
            self.clickedy=-1
        else:
            if (self.clickedy%2==0 and self.clickedx%2==0) or (self.clickedy%2==1 and self.clickedx%2==1):
                self.buttons[self.clickedy][self.clickedx].setStyleSheet("background-color: #bcaaa4")
            else:
                self.buttons[self.clickedy][self.clickedx].setStyleSheet("")
            self.buttons[self.clickedy][self.clickedx].setCheckable(False)
            self.clickedx=-1
            self.clickedy=-1
        self.app.processEvents()
        self.update()
        
        
if __name__ == '__main__':
    

    ex = Example()
    sys.exit(ex.app.exec_())
