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
        starting_state should be a list of the form ['F', '-', 'T']
        '''
        self.board_state = starting_state
        self.board_size = len(self.board_state)
