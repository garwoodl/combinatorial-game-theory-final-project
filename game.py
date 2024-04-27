'''
File to implement the game Toads and Frogs
'''


class ToadAndFrogs:
    '''
    Magnages game aspects such as
    - board state
    - verifying legal moves
    - updating with move input
    - checking for wins
    '''

    def __init__(self, starting_state: list[str]):
        '''
        Constructor for the ToadsAndFrogs class
        starting_state should be a list containing 1, 0, -1
        '''
        self.starting_state = starting_state
        self.current_state = self.starting_state
        self.board_size = len(self.current_state)
        self.marker_chars = {1: "T", 0: " ", -1: "F"}

    def reset(self):
        '''
        Reset the board to the starting position
        '''
        self.current_state = self.starting_state

    def __str__(self):
        '''
        return the current state as a string
        '''
        s = '|'
        for square in self.current_state:
            s += self.marker_chars[square] + '|'
        return s


def main():
    print("Hellow World")
    G = ToadAndFrogs([1, 0, 0, -1])
    print(G)


if __name__ == "__main__":
    main()
