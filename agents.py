from states import GameState, TOAD, FROG, BLANK
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

    def __init__(self, initial_state: GameState, amphibian=TOAD, agent_name=''):
        self.board_size = initial_state.board_size
        self.amphibian = amphibian
        self.agent_name = agent_name
        # make a Q-table later

    def __str__(self):
        s = ''
        s += f'agent_name: {self.agent_name} \n'
        s += f'board_size: {self.board_size} \n'
        s += f'amphibian: {self.amphibian} \n'
        return s

    def choose_move(self, state: GameState):
        '''
        This will be overrideen
        Any choose_move function should return an integer between 1 and
        the number of amphibians of that agent
        If there are no legal moves then it should return False
        '''
        pass


class RandomAgent(Agent):
    '''
    An agent that will always pick a random move
    from a state
    '''
    def __init__(self, initial_state: GameState, amphibian=TOAD, agent_name=''):
        Agent.__init__(self, initial_state, amphibian, agent_name)

    def choose_move(self, state: GameState):
        legal_moves = state.get_legal_moves()
        if len(legal_moves) == 0:
            return False
        return rand.choice(legal_moves)


class EndAgent(Agent):
    '''
    This agent will always pick the amphibian with legal move
    that is first or last depending on desire
    If 'type' is first it will pick first and if type is 'last'
    it will pick the last
    '''
    def __init__(self, initial_state: GameState, amphibian=TOAD, agent_name='', type='first'):
        self.type = str.lower(type)
        Agent.__init__(self, initial_state, amphibian, agent_name)

    def choose_move(self, state: GameState):
        legal_moves = state.get_legal_moves()
        if len(legal_moves) == 0:
            return False
        if self.type == 'first':
            return min(legal_moves)
        elif self.type == 'last':
            return max(legal_moves)
        else:
            raise ValueError('type must be either (first) or (last)')

