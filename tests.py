from states import GameState, TOAD, FROG, BLANK


def test_state():
    board = [1, 0, 0, -1]
    G = GameState(board)
    assert str(G) == '|T| | |F|'
    assert G.num_frogs == G.num_toads == 1
    assert G.toad_locs == [None, 0]
    assert G.frog_locs == [None, 3], G.frog_locs


def test_get_legal_moves():
    # G1 = GameState([1, 0, 0, -1], TOAD)
    # assert G1.get_legal_moves() == [1], G1.get_legal_moves()
    # G2 = GameState([1, -1, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1], TOAD)
    # assert G2.get_legal_moves() == [2, 3], f'{G2.get_legal_moves()}'
    # G3 = GameState([1, -1, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1], FROG)
    # assert G3.get_legal_moves() == [2, 5]

    G1 = GameState([1, 0, 0, -1], TOAD)
    assert G1.get_legal_moves() == set([1]), G1.get_legal_moves()
    G2 = GameState([1, -1, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1], TOAD)
    assert G2.get_legal_moves() == set([2, 3]), f'{G2.get_legal_moves()}'
    G3 = GameState([1, -1, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1], FROG)
    assert G3.get_legal_moves() == set([2, 5])


def test_make_move():
    G = GameState([1, 0, 0, -1])
    G2 = GameState([0, 1, 0, -1])
    G.make_move(1)
    assert G == G2, f"{G} not equal to {G2}"


def main():
    test_state()
    test_get_legal_moves()
    test_make_move()

    print("All tests passed!")


if __name__ == "__main__":
    main()
