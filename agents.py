from game import GameState, TOAD, FROG, BLANK
import random as rand


class Agent:
    '''
    Class to implement the various agents
    that learn to play Toads and Frogs. Each agent
    - sees a state
    - returns an action
    - an Agent object may have some previous training data
    such as a Q-table
    '''

    def __init__(self, initial_state: GameState, amphibian=TOAD):
        self.board_size = self.board_size
        self.amphibian = amphibian
        # make a Q-table later

    def __str__(self):
        s = ''
        s += f'board_size: {self.board_size} \n'
        s += f'amphibian: {self.amphibian} \n'
        return s

    def choose_move(self, state: GameState):
        '''
        This will be overrideen
        '''
        pass


class RandomAgent(Agent):
    '''
    An agent that will always pick a random move
    from a state
    '''
    def choose_move(self, state: GameState):
        return rand.choice(state.get_legal_moves(player=self.amphibian))
