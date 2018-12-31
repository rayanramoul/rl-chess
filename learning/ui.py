import sys
from board import board

from PyQt5.QtWidgets import (QWidget, QToolTip, 
    QPushButton, QApplication)
from PyQt5.QtGui import QFont    


class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        self.board=board()
        self.initUI()
        
        
    def initUI(self):
        
        QToolTip.setFont(QFont('SansSerif', 10))
        
        
        buttons=[[QPushButton('Button', self) for x in range(8)] for y in range(8)]
        a=10
        b=10
        indx=0
        indy=0
        for i in buttons:
            for j in i:
                try:
                    k="resources/icons/"+self.board.getpiece(indx,indy).id+".png"
                    import os
                    print("\tK : "+k)
                    for gr in os.listdir("resources/icons/"):
                        #print(gr)
                        if "resources/icons/"+gr==k:
                            print("Trouvé")
                    pixmap = QtGui.QPixmap("./resources/ray.jpg")
                    j.setIcon(QtGui.QIcon("./resources/ray.jpg"))
                    j.setStyleSheet("border-image:url(ray.jpg)")
                    j.setIconSize(QtCore.QSize(100,100))
                except:
                    j.setText("")
                j.resize(100,100)
                j.move(b, a)
                a=a+100
                indx=indx+1
            a=10
            indx=0
            b=b+200
            indy=indy+1       
        
        self.setGeometry(200, 300, 1024, 768)
        self.setWindowTitle('openChess')    
        self.show()
        
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
