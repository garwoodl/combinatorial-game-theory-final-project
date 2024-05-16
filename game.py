from agents import Agent, RandomAgent, EndAgent
from states import GameState, TOAD, FROG, BLANK
import numpy as np
import matplotlib.pyplot as plt
import time


def run_game_loop(initial_state: GameState, toad_agent: Agent,
                  frog_agent: Agent, verbose=True):
    '''
    Plays a game between toad_agent and frog_agent
    Returns True iff toad_agent wins
    '''
    if verbose:
        s = f"Initializing game between {toad_agent.agent_name} as Toads"
        s += f" and {frog_agent.agent_name} as Frogs..."
        print(s)
        print(initial_state)
    current_state = initial_state.copy()
    game_over = False
    move_count = 0
    while not game_over:
        toad_move = toad_agent.choose_move(current_state)
        if toad_move is False:
            game_over = False
            break
        current_state.make_move(move=toad_move)
        move_count += 1
        if verbose:
            print(f"{toad_agent.agent_name} makes move {toad_move}")
            print(current_state)

        frog_move = frog_agent.choose_move(current_state)
        if frog_move is False:
            game_over = False
            break
        current_state.make_move(move=frog_move)
        move_count += 1
        if verbose:
            print(f"{frog_agent.agent_name} makes move {frog_move}")
            print(current_state)

    if toad_move is False:  # toads had no moves
        if verbose:
            print(f"{frog_agent.agent_name} playing as Frogs wins in {move_count} moves!")
        return False
    elif frog_move is False:  # frogs had no moves
        if verbose:
            print(f"{toad_agent.agent_name} playing as Toads wins in {move_count} moves!")
        return True


def simulate_many_games(num_games: int, initial_state: GameState,
                        toad_agent: Agent, frog_agent: Agent, starting_player='mix', verbose=False):
    '''
    Run a simulation of many games and return the results aa numpy array
    of the form [TOAD, FROG, FROG, ...] giving the winner of each game
    starting_player defaults to 'mix' which means every game is alternated between who starts first
    if starting_player is given as TOAD or FROG then that amphibian will start all games
    '''
    results = np.zeros(num_games)
    for i in range(num_games):
        G = initial_state.copy()
        if starting_player == 'mix':
            G.current_player = TOAD if (i % 2 == 0) else FROG # alternate every other game
        else: 
            G.current_player = starting_player  # can be used to modify who starts the gamess
        result = run_game_loop(G, toad_agent, frog_agent, verbose=verbose)
        winner = TOAD if result else FROG
        results[i] = winner
    return results


def main():
    a = 5
    b = 10
    initial_position = [TOAD] * a + [BLANK] * b + [FROG] * a
    G = GameState(initial_position, starting_player=TOAD)
    agent1 = RandomAgent(initial_state=G, amphibian=TOAD, agent_name='random1')
    # agent1 = EndAgent(initial_state=G, amphibian=TOAD, agent_name='first', type='first')
    agent2 = RandomAgent(initial_state=G, amphibian=FROG, agent_name='random2')
    # run_game_loop(G, agent1, agent2, verbose=False)

    H = GameState(initial_position)
    num_games = 1000
    tic = time.time()
    results = simulate_many_games(num_games, H, agent1, agent2, verbose=False)
    toc = time.time()
    print(f'Simulation took {round(toc - tic, 5)} seconds')

    # Count the number of wins for each player
    t_wins = np.sum(results == TOAD)
    f_wins = np.sum(results == FROG)
    print(f"Toads won {t_wins} games ({round(t_wins / num_games * 100, 4)}%)")
    print(f"Frogs won {f_wins} games ({round(f_wins / num_games * 100, 4)}%)")

    # Plotting the results
    plt.bar(['Toad', 'Frog'], [t_wins, f_wins], color=['blue', 'red'])
    plt.xlabel('Player')
    plt.ylabel('Number of Wins')
    plt.title('Distribution of Wins')
    plt.show()


if __name__ == "__main__":
    main()
