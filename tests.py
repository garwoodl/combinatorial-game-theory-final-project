from game import GameState, TOAD, FROG, BLANK


def test_state():
    board = [1, 0, 0, -1]
    G = GameState(board)
    assert str(G) == '|T| | |F|'


def test_get_legal_moves():
    G1 = GameState([1, 0, 0, -1])
    assert G1.get_legal_moves(TOAD) == [1]
    assert G1.get_legal_moves(FROG) == [1]
    G2 = GameState([1, -1, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1])
    assert G2.get_legal_moves(TOAD) == [3, 2], f'{G2.get_legal_moves(TOAD)}'
    assert G2.get_legal_moves(FROG) == [2, 5]


def test_make_move():
    G = GameState([1, 0, 0, -1])
    G2 = GameState([0, 1, 0, -1])
    G.make_move(1, 1)
    assert G == G2


def main():
    test_state()
    test_get_legal_moves()
    test_make_move()

    print("All tests passed!")


if __name__ == "__main__":
    main()
