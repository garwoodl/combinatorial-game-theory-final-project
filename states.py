'''
File to implement the game Toads and Frogs
'''
import copy

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

    def __init__(self, starting_state: list[int], starting_player=TOAD):
        '''
        Constructor for the ToadsAndFrogs class
        starting_state should be a list containing 1, 0, -1
        toad_first says that the toad player should move first
        in this position (may not be important)
        '''
        self.current_player = starting_player
        self.starting_state = starting_state
        self.current_state = self.starting_state
        self.board_size = len(self.current_state)
        self.marker_chars = {TOAD: "T", BLANK: " ", FROG: "F"}

        self.num_toads = 0
        self.num_frogs = 0
        self.id = 0  # this is a unique id for each state. ternary with
        # BLANK = 0, TOAD = 1, FROG = 2
        for i, square in enumerate(self.current_state):
            if square == TOAD:
                self.id += 3 ** i
                self.num_toads += 1
            elif square == FROG:
                self.num_frogs += 1
                self.id += 2 * (3 ** i)

        # we will keep track of the amphibian's location from
        # their perspective as well the toad that is move k
        # has index in the state at toad_locs[move]
        self.toad_locs = [None]
        self.frog_locs = [None]
        for i, square in enumerate(self.current_state):
            if square == FROG:
                self.frog_locs.append(i)
        for i, square in reversed(list(enumerate(self.current_state))):
            if square == TOAD:
                self.toad_locs.append(i)

    def __str__(self):
        '''
        return the current state as a string
        '''
        s = '|'
        for square in self.current_state:
            s += self.marker_chars[square] + '|'
        return s

    def __eq__(self, s2):
        '''
        true if same board
        '''
        return self.current_state == s2.current_state

    def copy(self):
        # G = GameState(self.current_state, starting_player=self.current_player)
        # return G
        return copy.deepcopy(self)

    def reset(self):
        '''
        Reset the board to the starting position
        '''
        self.current_state = self.starting_state

    # def get_move_index(self, player, move: int):
    #     '''
    #     Converts the move integer into a list index
    #     O(n)
    #     '''
    #     if player == TOAD:
    #         x = 0 # last toad seen
    #         i = self.board_size - 1 # index in list
    #         while x < move:
    #             if self.current_state[i] == TOAD:
    #                 x += 1
    #             if x < move:
    #                 i -= 1
    #     elif player == FROG:
    #         x = 0
    #         i = 0
    #         while x < move:
    #             if self.current_state[i] == FROG:
    #                 x += 1
    #             if x < move:
    #                 i += 1
    #     return i

    # def is_move_legal(self, player, move: int):
        # get the move index
        # i = self.move_to_index(move)

        # # determine if its legal
        # if player == TOAD:
        #     if i+2 <= self.board_size - 1:
        #         if self.current_state[i+1] == FROG and self.current_state[i+2] == BLANK:
        #             return True
        #     if i+1 <= self.board_size - 1:
        #         if self.current_state[i+1] == BLANK:
        #             return True
        # elif player == FROG:
        #     if 0 <= i-2:
        #         if self.current_state[i-1] == TOAD and self.current_state[i-2] == BLANK:
        #             return True
        #     if 0 <= i-1:
        #         if self.current_state[i-1] == BLANK:
        #             return True
        # return False

    def get_legal_moves(self):
        '''
        A move is a tuple (player, amphibean_index)
        giving which player the move is for and which
        of their amphibeans to move (counting right to
        left for TOAD and left to right for FROG)
        '''
        legal_moves = []
        amphibian_num = 1
        if self.current_player == TOAD:
            for i, square in reversed(list(enumerate(self.current_state))):  # moves right to left
                if square == TOAD:
                    if i+2 <= self.board_size - 1:
                        if self.current_state[i+1] == FROG and self.current_state[i+2] == BLANK:
                            legal_moves.append(amphibian_num)
                    if i+1 <= self.board_size - 1:
                        if self.current_state[i+1] == BLANK:
                            legal_moves.append(amphibian_num)
                    amphibian_num += 1
        elif self.current_player == FROG:
            for i, square in enumerate(self.current_state): # moves right to left
                if square == FROG:
                    if 0 <= i-2:
                        if self.current_state[i-1] == TOAD and self.current_state[i-2] == BLANK:
                            legal_moves.append(amphibian_num)
                    if 0 <= i-1:
                        if self.current_state[i-1] == BLANK:
                            legal_moves.append(amphibian_num)
                    amphibian_num += 1
        return legal_moves

    def make_move(self, move: int):
        '''
        player should be either TOAD or FROG
        move should be the nth frog from the middle going
        out to the respective starting position
        Returns False if the move is illegal
        otherwise returns True
        '''
        if self.current_player == TOAD:
            idx = self.toad_locs[move]
            if idx + 1 <= self.board_size - 1:
                if self.current_state[idx + 1] == BLANK:
                    # slide forward
                    self.current_state[idx] = BLANK
                    self.current_state[idx + 1] = TOAD
                    self.toad_locs[move] += 1
                elif self.current_state[idx + 1] == FROG and idx + 2 <= self.board_size - 1:
                    if self.current_state[idx + 2] == BLANK:
                        # jump over frog
                        self.current_state[idx] = BLANK
                        self.current_state[idx + 2] = TOAD
                        self.toad_locs[move] += 2
        elif self.current_player == FROG:
            idx = self.frog_locs[move]
            if idx - 1 >= 0:
                if self.current_state[idx - 1] == BLANK:
                    # slide forward
                    self.current_state[idx] = BLANK
                    self.current_state[idx - 1] = FROG
                    self.frog_locs[move] -= 1
                elif self.current_state[idx - 1] == TOAD and idx - 2 >= 0:
                    if self.current_state[idx - 2] == BLANK:
                        # jump over toad
                        self.current_state[idx] = BLANK
                        self.current_state[idx - 2] = FROG
                        self.frog_locs[move] -= 2

        # set current player to the other player
        if self.current_player == TOAD:
            self.current_player = FROG
        elif self.current_player == FROG:
            self.current_player = TOAD
        return self


def main():
    G = GameState([1, -1, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1])
    print(G)
    print(G.get_legal_moves(TOAD))


if __name__ == "__main__":
    main()
