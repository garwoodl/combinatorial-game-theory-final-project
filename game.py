from agents import Agent, RandomAgent
from states import GameState, TOAD, FROG, BLANK


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
            print(f"{toad_agent.agent_name} makes move {move_count}")
            print(current_state)

        print("TEST", current_state.current_player)
        frog_move = frog_agent.choose_move(current_state)
        if frog_move is False:
            game_over = False
            break
        current_state.make_move(move=frog_move)
        move_count += 1
        if verbose:
            print(f"{frog_agent.agent_name} makes move {move_count}")
            print(current_state)
        print("TEST2", current_state.current_player)

    if toad_move is False:  # toads had no moves
        if verbose:
            print(f"{frog_agent.agent_name} playing as Frogs wins!")
        return False
    elif frog_move is False:  # frogs had no moves
        if verbose:
            print(f"{toad_agent.agent_name} playing as Toads wins!")
        return True


def main():
    G = GameState([1, 0, 0, -1])
    agent1 = RandomAgent(initial_state=G, amphibian=TOAD, agent_name='random1')
    agent2 = RandomAgent(initial_state=G, amphibian=FROG, agent_name='random2')
    run_game_loop(G, agent1, agent2)


if __name__ == "__main__":
    main()
