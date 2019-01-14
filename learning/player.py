from random import *
class Player():
    def __init__(self, side):
        self.side=side
    
    def choose(self, moves):
        size=len(moves)
        return randint(0,size-1)