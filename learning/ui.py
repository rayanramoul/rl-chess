import sys
from board import board
from functools import partial
from PyQt5.QtWidgets import (QWidget, QToolTip, 
    QPushButton, QApplication, QLabel, QListWidget, QListWidgetItem)
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import QSize, QThread, pyqtSignal
from player import Player
from time import sleep



   
class AutoThread(QThread):
    def __init__(self, thing):
        QThread.__init__(self)
        self.thing=thing
    def run(self):
        print("CREAAAAAAAAATE")
        while self.thing.autoplay.isChecked():
            print("MMMMM 1")
            self.thing.board.allmoves()
            self.thing.possmov=self.thing.board.moves
            if self.thing.board.turn=="white":
                k=self.thing.pwhite.choose(self.thing.possmov)
            else:
                k=self.thing.pblack.choose(self.thing.possmov)
            el=self.thing.possmov[k]
            print("MOVE CHOISIIIIIIII : ")
            el.describe()
            self.thing.board.move(el.basex, el.basey, el.newx, el.newy)
            self.thing.redraw(el.basex, el.basey, el.newx, el.newy)
            sleep(4)
            print("MMMMM 2")
            self.thing.update()
            print("NEW BOARD /")
            self.thing.board.print()
            self.thing.app.processEvents()

 





class Example(QWidget):
    def __init__(self):
        self.app = QApplication(sys.argv)
        super().__init__()
        self.board=board()
        self.initUI()
        self.pwhite=Player("white")
        self.pblack=Player("black")
        
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
        self.state=QLabel(self)
        self.state.move(850,750)
        self.state.setText("No check")
        self.state.resize(200, 20)
        self.state.setFont(QFont('SansSerif', 25))

        self.possiblemoves=QListWidget(self)
        self.possiblemoves.move(830,100)
        self.possiblemoves.resize(180, 500)
        
        self.lastbasex=0
        self.lastbasey=0
        self.lastnewx=0
        self.lastnewy=0
        self.possiblemoves.clear()
        self.possmov=self.board.moves
        self.autoplay=QPushButton(self)

        self.autoplay.setText("AUTOPLAY")
        self.autoplay.setCheckable(True)
        self.autoplay.move(850,700)
        self.autoplay.resize(100,30)
        self.autoplay.clicked.connect(self.createAutoPlay)
        self.autothread=AutoThread(self)

        self.retur=QPushButton(self)
        self.retur.move(850,650)
        self.retur.resize(100,30)
        self.retur.setText("RETURN")
        self.retur.clicked.connect(self.revoke)

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

    def createAutoPlay(self):
        self.autothread.start()

    def revoke(self):
        for i in self.board.revoke():
            if self.board.getpiece(i.basex,i.basey)!=None:   
                k="resources/icons/"+self.board.getpiece(i.basex,i.basey).id+".png"
                pixmap = QPixmap(k)
                self.buttons[i.basey][i.basex].setText("")
                self.buttons[i.basey][i.basex].setIcon(QIcon(pixmap))
                self.buttons[i.basey][i.basex].setIconSize(QSize(100,100))
                self.buttons[i.basey][i.basex].setText("")
                self.buttons[i.basey][i.basex].setCheckable(False)
                self.buttons[i.basey][i.basex].update()
            if self.board.getpiece(i.newx,i.newy)!=None:   
                k="resources/icons/"+self.board.getpiece(i.newx,i.newy).id+".png"
                pixmap = QPixmap(k)
                self.buttons[i.newy][i.newx].setText("")
                self.buttons[i.newy][i.newx].setIcon(QIcon(pixmap))
                self.buttons[i.newy][i.newx].setIconSize(QSize(100,100))
                self.buttons[i.newy][i.newx].setText("")
                self.buttons[i.newy][i.newx].setCheckable(False)
                self.buttons[i.newy][i.newx].update()
            if self.board.getpiece(i.basex,i.basey)==None:
                pixmap = QPixmap()
                self.buttons[i.basey][i.basex].setText("")
                self.buttons[i.basey][i.basex].setIcon(QIcon(pixmap))
                self.buttons[i.basey][i.basex].setIconSize(QSize(100,100))
                self.buttons[i.basey][i.basex].setText("")
                self.buttons[i.basey][i.basex].setCheckable(False)
                self.buttons[i.basey][i.basex].update()
            if self.board.getpiece(i.newx,i.newy)==None:
                pixmap = QPixmap()
                self.buttons[i.newy][i.newx].setText("")
                self.buttons[i.newy][i.newx].setIcon(QIcon(pixmap))
                self.buttons[i.newy][i.newx].setIconSize(QSize(100,100))
                self.buttons[i.newy][i.newx].setText("")
                self.buttons[i.newy][i.newx].setCheckable(False)
                self.buttons[i.newy][i.newx].update()
        bim=str(self.board.check())
        if bim!="":
            self.state.setText(bim+" in check.")
        else:
            self.state.setText("No Check")
        self.turn.setText(self.board.turn+"'s turn")
        self.possiblemoves.clear()
        self.possmov=self.board.moves
        for i in self.possmov:
            item = QListWidgetItem(i.string())
            self.possiblemoves.addItem(item)
#            self.possiblemoves.repaint()
        self.possiblemoves.update()
        self.update()

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
        print("redraw")
        self.possiblemoves.clear()
        self.possmov=self.board.moves
        for i in self.possmov:
            item = QListWidgetItem(i.string())
            self.possiblemoves.addItem(item)
#            self.possiblemoves.repaint()
        self.possiblemoves.update()

        if self.board.getpiece(beginx,beginy)!=None:
            
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
        bim=str(self.board.check())
        if bim!="":
            self.state.setText(bim+" in check.")
        else:
            self.state.setText("No Check")
        self.turn.setText(self.board.turn+"'s turn")
        self.update()
    def handleClickCase(self,indx,indy):
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
