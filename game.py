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
        self.id = 0 # this is a unique id for each state. ternary with 
        # BLANK = 0, TOAD = 1, FROG = 2
        for square, i in enumerate(self.current_state):
            if square == TOAD:
                self.id += 3 ** i
                self.num_toads += 1
            elif square == FROG:
                self.num_frogs += 1
                self.id += 2 * (3 ** i)

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

    def move_to_index(self, player, move: int):
        '''
        Converts the move integer into a list index
        O(n)
        '''
        if player == TOAD:
            x = 0 # last toad seen
            i = self.board_size - 1 # index in list
            while x < move:
                if self.current_state[i] == TOAD:
                    x += 1
                if x < move:
                    i -= 1
        elif player == FROG:
            x = 0
            i = 0
            while x < move:
                if self.current_state[i] == FROG:
                    x += 1
                if x < move:
                    i += 1
        return i

    def is_move_legal(self, player, move: int):
        # get the move index
        i = self.move_to_index(move)

        # determine if its legal
        if player == TOAD:
            if i+2 <= self.board_size - 1:
                if self.current_state[i+1] == FROG and self.current_state[i+2] == BLANK:
                    return True
            if i+1 <= self.board_size - 1:
                if self.current_state[i+1] == BLANK:
                    return True
        elif player == FROG:
            if 0 <= i-2:
                if self.current_state[i-1] == TOAD and self.current_state[i-2] == BLANK:
                    return True
            if 0 <= i-1:
                if self.current_state[i-1] == BLANK:
                    return True
        return False

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

    def make_move(self, player, move: int):
        '''
        player should be either TOAD or FROG
        move should be the nth frog from the middle going
        out to the respective starting position
        Returns False if the move is illegal
        otherwise returns True
        '''
        if not self.is_move_legal(player, move):
            return False
        
        i = self.get_move_index(player, move)

        # make the move
        if player == TOAD:
            if i+2 <= self.board_size - 1:
                if self.current_state[i+1] == FROG and self.current_state[i+2] == BLANK:
                    self.current_state[i] = BLANK
                    self.current_state[i+2] = TOAD
            if i+1 <= self.board_size - 1:
                if self.current_state[i+1] == BLANK:
                    self.current_state[i] = BLANK
                    self.current_state[i+1] = TOAD
        elif player == FROG:
            if 0 <= i-2:
                if self.current_state[i-1] == TOAD and self.current_state[i-2] == BLANK:
                    self.current_state[i] = BLANK
                    self.current_state[i-2] = FROG
            if 0 <= i-1:
                if self.current_state[i-1] == BLANK:
                    self.current_state[i] = BLANK
                    self.current_state[i-1] = FROG
        return True


def run_game_loop(initial_state):
    game_over = False
    while not game_over:
        ...


def main():
    G = GameState([1, -1, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1])
    print(G)
    print(G.get_legal_moves(TOAD))


if __name__ == "__main__":
    main()
