from game import *


class Agent:
    '''
    Class to implement the various agents
    that learn to play Toads and Frogs. Each agent
    - sees a state
    - returns an action
    - an Agent object may have some previous training data
    such as a Q-table
    '''
    
    def __init__(self, board_size, amphibian=TOAD):
        self.board_size = self.board_size
        self.amphibian = amphibian
        # make a Q-table later
