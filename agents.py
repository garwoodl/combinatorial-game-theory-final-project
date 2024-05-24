from states import GameState, TOAD, FROG, BLANK
import random as rand
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


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
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*rand.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class RLAgent(Agent):
    '''
    The agent that will learn to play Toads and Frogs through Q-learning
    '''
    def __init__(self, initial_state: GameState, amphibian=TOAD, agent_name='rl',
                 lr=1e-3, batch_size=10, buffer_capacity=1000):
        self.lr = lr
        self.gamma = 0.9
        self.buffer_capacity = buffer_capacity
        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.target_update_freq = 5  # how often to update the target network
        Agent.__init__(self, initial_state, amphibian, agent_name=agent_name)
        self.model = self.initialize_model()
        self.target_model = self.initialize_model()
        self.update_target_network()
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

    def update_target_network(self):
        '''
        Copy the main parameters to the target network
        '''
        self.target_model.load_state_dict(self.model.state_dict())

    def state_to_q_vals(self, state: GameState):
        '''
        Returns a tensor of all q values for all actions in a given state
        '''
        vec = torch.tensor(state.current_state, dtype=torch.float32).unsqueeze(0)
        return self.model(vec)

    def target_state_to_q_vals(self, state: GameState):
        vec = torch.tensor(state.current_state, dtype=torch.float32).unsqueeze(0)
        return self.target_model(vec)

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

    def train(self, opponent: Agent, num_episodes: int, start_epsilon=0, end_epsilon=0, verbose=True):
        '''
        The main train loop that runs for num_episodes
        After each state transfer (move in any episode) the
        resulting state, reward, and done will be pushed to
        the replay memory for training.
        Plays against the Agent given.
        Smoothly decays epsilon from start to end (I chose inverse sqrt)
        '''
        epsilon = start_epsilon  # set up epsilon schedule later
        losses = []
        for episode in range(num_episodes):
            if verbose:
                if episode % 200 == 0:
                    print(f"Training {round(100 * episode / num_episodes, 2)}% complete...")
            state = self.initial_state.copy()
            episode_done = False
            # decay epsilon from start to end using inverse sqrt
            epsilon = end_epsilon + (start_epsilon - end_epsilon) / (episode + 1) ** 0.5 
            while not episode_done:
                action = self.choose_move_train(state, epsilon)
                next_state, reward, done = self.step(state, action)
                # print("State", state)
                # print("Action", action)
                # print("Reward", reward)
                # print("Next State", next_state)

                # the opponent responds
                if not done:
                    opp_action = opponent.choose_move(next_state)
                    # this reward is still with respect to the agent being trained
                    next_state, opp_reward, done = self.step(next_state, opp_action)
                    reward += opp_reward # this is boring for most movese but technically correct
                    # if we later implement incremental awards

                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_done = done

                if len(self.buffer) >= self.batch_size:
                    loss = self.optimize_model()  # this optimizes and returns the loss
                    losses.append(loss)

            if episode % self.target_update_freq == 0:
                self.update_target_network()
        if verbose:
            print("Training finished!")
        return losses

    def optimize_model(self):
        '''
        Using the current replay buffer sample a batch to train the
        main network on.
        Returns the loss for this optimization
        '''
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        state_batch = torch.stack([torch.tensor(s.current_state, dtype=torch.float32) for s in states])
        action_batch = torch.tensor(actions) - 1  # zero-indexed
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool)
        non_final_next_states = torch.stack([torch.tensor(s.current_state, dtype=torch.float32) for s, d in zip(next_states, dones) if not d])

        current_q_vals = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        next_q_vals = torch.zeros(self.batch_size)
        next_q_vals[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        expected_q_vals = reward_batch + (self.gamma * next_q_vals)

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q_vals, expected_q_vals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_agent_model(filename=''):
        '''
        Save the agent parameters to a file in the directory
        '''
        ...


def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss over time')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    a = 2
    b = 2
    initial_position = [TOAD] * a + [BLANK] * b + [FROG] * a
    G = GameState(initial_position, starting_player=TOAD)
    agent1 = RLAgent(G, TOAD, batch_size=50)
    agent2 = RandomAgent(G, FROG)
    losses = agent1.train(opponent=agent2, num_episodes=1000, start_epsilon=0.1)
    plot_losses(losses)


if __name__ == "__main__":
    main()
