'''
File to implement the game Toads and Frogs
'''

TOAD = 1
FROG = -1
BLANK = 0


class GameState:
    '''
    Magnages game aspects such as
    - board state
    - verifying legal moves
    - updating with move input
    - checking for wins
    '''

    def __init__(self, starting_state: list[str], current_player=TOAD):
        '''
        Constructor for the ToadsAndFrogs class
        starting_state should be a list containing 1, 0, -1
        toad_first says that the toad player should move first 
        in this position (may not be important)
        '''
        self.current_player = current_player
        self.starting_state = starting_state
        self.current_state = self.starting_state
        self.board_size = len(self.current_state)
        self.marker_chars = {TOAD: "T", BLANK: " ", FROG: "F"}

        self.num_toads = 0
        self.num_frogs = 0
        for square in self.current_state:
            if square == TOAD:
                self.num_toads += 1
            elif square == FROG:
                self.num_frogs += 1

    def __str__(self):
        '''
        return the current state as a string
        '''
        s = '|'
        for square in self.current_state:
            s += self.marker_chars[square] + '|'
        return s

    def reset(self):
        '''
        Reset the board to the starting position
        '''
        self.current_state = self.starting_state

    def get_legal_moves(self, player=TOAD):
        '''
        A move is a tuple (player, amphibean_index)
        giving which player the move is for and which
        of their amphibeans to move (counting right to
        left for TOAD and left to right for FROG)
        '''
        legal_moves = []
        if player == TOAD:
            # because we are looking left to right in the for loop
            amphibian_num = self.num_toads
        else:
            amphibian_num = 1

        for i, square in enumerate(self.current_state):
            if player == square == TOAD:
                if i+2 <= self.board_size - 1:
                    if self.current_state[i+1] == FROG and self.current_state[i+2] == BLANK:
                        legal_moves.append(amphibian_num)
                if i+1 <= self.board_size - 1:
                    if self.current_state[i+1] == BLANK:
                        legal_moves.append(amphibian_num)
                amphibian_num -= 1
            elif player == square == FROG:
                if 0 <= i-2:
                    if self.current_state[i-1] == TOAD and self.current_state[i-2] == BLANK:
                        legal_moves.append(amphibian_num)
                if 0 <= i-1:
                    if self.current_state[i-1] == BLANK:
                        legal_moves.append(amphibian_num)
                amphibian_num += 1
        return legal_moves


def main():
    G = GameState([1, -1, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1])
    print(G)
    print(G.get_legal_moves(TOAD))


if __name__ == "__main__":
    main()
