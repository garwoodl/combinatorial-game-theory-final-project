from states import GameState, TOAD, FROG, BLANK
import random as rand
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch


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

# memory class from Pytorch DQN tutorial
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# from collections import namedtuple, deque
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))


# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return rand.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)


class RLAgent(Agent):
    '''
    The agent that will learn to play Toads and Frogs through Q-learning
    '''
    def __init__(self, initial_state: GameState, amphibian=TOAD, agent_name='rl', lr=1e-3, batch_size=10):
        self.lr = lr
        self.gamma = 0.9
        Agent.__init__(self, initial_state, amphibian, agent_name=agent_name)
        self.model = self.initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.batch_size = batch_size
        self.rewards = {
            'win': 10,
            'loss': -10,
            'illegal': -15
        }

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

    def state_to_q_vals(self, state: GameState):
        '''
        Returns a tensor of all q values for all actions in a given state
        '''
        vec = torch.tensor(state.current_state, dtype=torch.float32).unsqueeze(0)
        return self.model(vec)

    def choose_move(self, state: GameState, epsilon=0):
        '''
        This is the 'nice' choose_move function to be used
        as the agent's outward facing function
        '''
        legal_moves = state.get_legal_moves()
        if len(legal_moves) == 0:
            return False

        with torch.no_grad():
            q_vals = self.state_to_q_vals(state)
            move = int(np.argmax(q_vals) + 1)
            if move in legal_moves:
                return move  # undo zero index
            else:
                return rand.choice(list(legal_moves))            

    def choose_move_train(self, state: GameState, epsilon=0):
        '''
        the choose move function during training
        can output illegal moves but will always give an integer
        epsilon-greedy
        '''
        if rand.random() < epsilon:
            return rand.choice(range(1, self.num_moves + 1))
        else:
            with torch.no_grad():
                q_vals = self.state_to_q_vals(state)
                move = int(np.argmax(q_vals) + 1)
                # print(q_vals)
                # print(move)
                return move

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

        G.make_move(action)
        game_over, winner = G.is_game_over()
        if game_over:
            if winner == self.amphibian:
                return G, self.rewards['win'], True
            else:
                return G, self.rewards['loss'], True
        else:
            return G, 0, False

    def train(self, num_episodes: int, epsilon=0):
        steps_done = 0
        losses = []
        for epsiode in range(num_episodes):
            state = self.initial_state.copy()
            target_vals = []
            actual_vals = []
            episode_done = False
            while not episode_done:
                # get the transition outcome
                action = self.choose_move_train(state, epsilon)
                next_state, reward, done = self.step(state, action)

                # the actual q value we see is what the model evaluates rn for this action
                current_q_vals = self.state_to_q_vals(state)
                actual_q_val = current_q_vals[0, action - 1]
                print('current_q_vals', current_q_vals)
                print('actual_q_val', actual_q_val)
                actual_vals.append(actual_q_val)

                # bellman equation
                # y is the target q value for the state
                if done:
                    target_q_val = torch.tensor(reward, dtype=torch.float32)
                else:
                    next_q_vals = self.state_to_q_vals(next_state)
                    max_next_q_val = next_q_vals.max().item()
                    target_q_val = torch.tensor(reward + self.gamma * max_next_q_val, dtype=torch.float32)
                target_vals.append(target_q_val.detach())

                state = next_state
                episode_done = done
                steps_done += 1
                print(actual_vals)
                if len(actual_vals) > self.batch_size:
                    actual_val_batch = torch.stack(actual_vals[-self.batch_size:])
                    target_val_batch = torch.stack(target_vals[-self.batch_size:])

                    loss_fn = nn.SmoothL1Loss()
                    loss = loss_fn(actual_val_batch, target_val_batch)
                    losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        return losses


    # def optimize_model(self):
    #     if len(self.memory) < self.batch_size:
    #         return
    #     transitions = self.memory.sample(self.batch_size)
    #     batch = Transition(*zip(*transitions))  # reorders the Transition objects
    #     not_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                             batch.next_state)))
    #     not_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    #     state_batch = torch.cat(batch.state)
    #     action_batch = torch.cat(batch.action)
    #     reward_batch = torch.cat(batch.reward)

    #     # find the q values for the actions that were taken
    #     state_action_values = self.model(state_batch).gather(1, action_batch)

    #     next_state_values = torch.zeros(self.batch_size)
    #     with torch.no_grad():
    #         next_state_values[not_final_mask] = 