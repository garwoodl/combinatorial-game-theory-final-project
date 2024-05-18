from states import GameState, TOAD, FROG, BLANK
import random as rand
import numpy as np
import torch.nn as nn


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
        self.initial_state = initial_state
        self.board_size = initial_state.board_size
        self.amphibian = amphibian
        self.agent_name = agent_name
        if self.amphibian == TOAD:
            self.num_moves = initial_state.num_toads
        elif self.amphibian == FROG:
            self.num_moves = initial_state.num_frogs

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
    def __init__(self, initial_state: GameState, amphibian=TOAD, agent_name='random'):
        Agent.__init__(self, initial_state, amphibian, agent_name)

    def choose_move(self, state: GameState):
        legal_moves = state.get_legal_moves()
        if len(legal_moves) == 0:
            return False
        return rand.choice(list(legal_moves))


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


class HumanInput(Agent):
    '''
    An object that handles human input from the command line
    '''
    def __init__(self, initial_state: GameState, amphibian=TOAD, agent_name='human'):
        Agent.__init__(self, initial_state, amphibian, agent_name)

    def choose_move(self, state: GameState):
        legal_moves = state.get_legal_moves()
        if len(legal_moves) == 0:
            return False

        move = input(f'{self.agent_name}, please enter a move: ')
        while True:
            try:
                move = int(move)
                if move in legal_moves:
                    return move
                else:
                    move = input(f'Illegal move. Must be in {list(legal_moves)}. Try again: ')
            except:
                move = input(f'Move must be an integer in {list(legal_moves)}. Try again: ')


class RLAgent(Agent):
    '''
    The agent that will learn to play Toads and Frogs through Q-learning
    '''
    def __init__(self, initial_state: GameState, amphibian=TOAD, agent_name='rl', lr=1e-3):
        self.model = self.initialize_model()
        self.rewards = {
            'win': 10,
            'loss': -10,
            'illegal': -15
        }
        self.lr = lr
        self.gamma = 0.9  # see if we need this later
        Agent.__init__(self, initial_state, amphibian)

    def initialize_model(self):
        '''
        Make the neural network for Q-function
        '''
        input_size = self.board_size
        h1 = 100
        h2 = 75
        output_size = self.num_moves
        model = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_size)
        )
        return model

    def choose_move(self, state: GameState, epsilon=0):
        '''
        Epsilon-greedy algorithm
        With probability epsilon, the agent will pick a random legal move
        With proability 1 - epsilon, the agent will pick the best action
        according to its models highest q_value
        '''
        if rand.random() < epsilon:
            return rand.choice(list(state.get_legal_moves()))
        else:
            q_vals = self.model(state)
            return np.argmax(q_vals) + 1  # undo zero index

    def step(self, state: GameState, action: int):
        '''
        Returns (next_state, reward, done)
        done means the episode is done
        If the move is illegal next_state will be None and done will be True
        '''
        G = state.copy()
        legal_moves = G.get_legal_moves()

        if action not in legal_moves:
            return None, self.rewards['illegal'], True

        next_state = G.make_move(action)
        game_over, winner = next_state.is_game_over()
        if game_over:
            if winner == self.amphibian:
                return next_state, self.rewards['win'], True
            else:
                return next_state, self.rewards['loss'], True
        else:
            return next_state, 0, False

    def train(self, num_episodes: int, epsilon=0):
        steps_done = 0
        epsilon = 0.1
        for epsiode in range(num_episodes):
            state = self.initial_state.copy()
            total_reward = 0
            episode_done = False
            while not episode_done:
                action = self.choose_move(state, epsilon)
                next_state, reward, done = self.step(state, action)
                total_reward += reward
                ## continue this later
