from board import board

b=board()
b.print()
print("\nTHE FORMAT TO MOVE : InitialX,InitialY,DestionationX,DestinationY\nExample :1,0,2,0\n")
while True:
    print(b.turn+"'s turn !")
    d=input(":")   
    print(d)
    l=[]
    l=d.split(",")
    k=[]
    for i in l:
        k.append(int(i))
    b.move(k[0],k[1],k[2],k[3])