from game import ToadAndFrogs


def test_state():
    board = [1, 0, 0, -1]
    G = ToadAndFrogs(board)
    assert str(G) == '|T| | |F|'


def main():
    test_state()

    print("All tests passed!")


if __name__ == "__main__":
    main()