"""Tests for board representation."""

from catan.board import Board
from catan.resources import Terrain


def test_standard_board_has_19_hexes():
    board = Board.standard()
    assert len(board.hexes) == 19


def test_standard_board_has_one_desert():
    board = Board.standard()
    deserts = [h for h in board.hexes if h.terrain == Terrain.DESERT]
    assert len(deserts) == 1


def test_desert_has_no_token():
    board = Board.standard()
    desert = next(h for h in board.hexes if h.terrain == Terrain.DESERT)
    assert desert.token is None
