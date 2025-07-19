import unittest

from engine.state.game_state import GameState
from lib.interact.map import Map
from lib.interact.tile import Tile


class TestPlaceTile(unittest.TestCase):
    def setUp(self) -> None:
        self.state = GameState()
        self.state.map = Map()

        # Helper for moving in grid
        self.DIRECTION_OFFSET = {
            "top_edge": (0, -1),
            "bottom_edge": (0, 1),
            "left_edge": (-1, 0),
            "right_edge": (1, 0),
        }

        # Helper for finding opposite edge
        self.OPPOSITE_EDGE = {
            "top_edge": "bottom_edge",
            "bottom_edge": "top_edge",
            "left_edge": "right_edge",
            "right_edge": "left_edge",
        }

    def tearDown(self) -> None:
        return super().tearDown()

    def test_basic_road_traversal(self) -> None:
        self.state.start_base_phase()
        U1 = self.state.map.get_tile_by_type("U", pop=True)
        U2 = self.state.map.get_tile_by_type("U", pop=True)
        U3 = self.state.map.get_tile_by_type("U", pop=True)

        pos1 = (85, 85)
        pos2 = (85, 84)
        pos3 = (85, 83)

        U1.placed_pos = pos1
        U2.placed_pos = pos2
        U3.placed_pos = pos3

        self.state.map._grid[pos1[1]][pos1[0]] = U1
        self.state.map._grid[pos2[1]][pos2[0]] = U2
        self.state.map._grid[pos3[1]][pos3[0]] = U3

        # We want all edges of all tiles
        assert set(self.state._traverse_connected_component(U1, "top_edge")) == set(
            [
                (U1, "top_edge"),
                (U2, "top_edge"),
                (U3, "top_edge"),
                (U1, "bottom_edge"),
                (U2, "bottom_edge"),
                (U3, "bottom_edge"),
            ]
        )

    def test_basic_city_traversal(self) -> None:
        self.state.start_base_phase()
        H1 = self.state.map.get_tile_by_type("H", pop=True)
        F1 = self.state.map.get_tile_by_type("F", pop=True)
        F2 = self.state.map.get_tile_by_type("F", pop=True)

        pos1 = (85, 85)
        pos2 = (86, 85)
        pos3 = (87, 85)

        H1.placed_pos = pos1
        F1.placed_pos = pos2
        F2.placed_pos = pos3

        self.state.map._grid[pos1[1]][pos1[0]] = H1
        self.state.map._grid[pos2[1]][pos2[0]] = F1
        self.state.map._grid[pos3[1]][pos3[0]] = F2

        assert set(self.state._traverse_connected_component(H1, "right_edge")) == set(
            [
                (H1, "right_edge"),
                (F1, "left_edge"),
                (F1, "right_edge"),
                (F2, "left_edge"),
                (F2, "right_edge"),
            ]
        )

        assert not self.state.get_completed_components(H1)

    @unittest.skip("Not implmented")
    def test_basic_monastarty(self) -> None:
        pass

    def test_complex_road_traversal(self) -> None:
        self.state.start_base_phase()
        start = self.state.map.get_tile_by_type("A", pop=True)
        start.rotate_clockwise(2)
        start.placed_pos = (85, 85)
        self.state.map._grid[85][85] = start

        expected_set: set[tuple["Tile", str]] = set()
        expected_set.add((start, "top_edge"))

        # K1
        K1 = self.state.map.get_tile_by_type("K", pop=True)
        k1_pos = (85, 84)
        K1.rotate_clockwise(2)
        K1.placed_pos = k1_pos
        self.state.map._grid[k1_pos[1]][k1_pos[0]] = K1
        expected_set.add((K1, "bottom_edge"))
        expected_set.add((K1, "right_edge"))
        assert not self.state.get_completed_components(start)

        # K2
        K2 = self.state.map.get_tile_by_type("K", pop=True)
        k2_pos = (86, 84)
        K2.placed_pos = k2_pos
        self.state.map._grid[k2_pos[1]][k2_pos[0]] = K2
        expected_set.add((K2, "left_edge"))
        expected_set.add((K2, "top_edge"))
        assert not self.state.get_completed_components(start)

        # U1
        U1 = self.state.map.get_tile_by_type("U", pop=True)
        u1_pos = (86, 83)
        U1.placed_pos = u1_pos
        self.state.map._grid[u1_pos[1]][u1_pos[0]] = U1
        expected_set.add((U1, "bottom_edge"))
        expected_set.add((U1, "top_edge"))
        assert not self.state.get_completed_components(start)

        # P1
        P1 = self.state.map.get_tile_by_type("P", pop=True)
        p1_pos = (86, 82)
        P1.placed_pos = p1_pos
        self.state.map._grid[p1_pos[1]][p1_pos[0]] = P1
        expected_set.add((P1, "bottom_edge"))
        expected_set.add((P1, "right_edge"))
        assert not self.state.get_completed_components(start)

        # U2
        U2 = self.state.map.get_tile_by_type("U", pop=True)
        u2_pos = (87, 82)
        U2.rotate_clockwise(1)
        U2.placed_pos = u2_pos
        self.state.map._grid[u2_pos[1]][u2_pos[0]] = U2
        expected_set.add((U2, "right_edge"))
        expected_set.add((U2, "left_edge"))
        assert not self.state.get_completed_components(start)

        # J1
        J1 = self.state.map.get_tile_by_type("J", pop=True)
        j1_pos = (88, 82)
        J1.rotate_clockwise(2)
        J1.placed_pos = j1_pos
        self.state.map._grid[j1_pos[1]][j1_pos[0]] = J1
        expected_set.add((J1, "left_edge"))
        expected_set.add((J1, "top_edge"))
        assert not self.state.get_completed_components(start)

        # U3
        U3 = self.state.map.get_tile_by_type("U", pop=True)
        u3_pos = (88, 81)
        U3.placed_pos = u3_pos
        self.state.map._grid[u3_pos[1]][u3_pos[0]] = U3
        expected_set.add((U3, "bottom_edge"))
        expected_set.add((U3, "top_edge"))
        assert not self.state.get_completed_components(start)

        # J2
        J2 = self.state.map.get_tile_by_type("J", pop=True)
        j2_pos = (88, 80)
        J2.placed_pos = j2_pos
        self.state.map._grid[j2_pos[1]][j2_pos[0]] = J2
        expected_set.add((J2, "bottom_edge"))
        expected_set.add((J2, "right_edge"))
        assert not self.state.get_completed_components(start)

        # U4
        U4 = self.state.map.get_tile_by_type("U", pop=True)
        u4_pos = (89, 80)
        U4.rotate_clockwise(1)
        U4.placed_pos = u4_pos
        self.state.map._grid[u4_pos[1]][u4_pos[0]] = U4
        expected_set.add((U4, "right_edge"))
        expected_set.add((U4, "left_edge"))
        assert not self.state.get_completed_components(start)

        # End tile
        end = self.state.map.get_tile_by_type("A", pop=True)
        end_pos = (90, 80)
        end.rotate_clockwise(1)
        end.placed_pos = end_pos
        self.state.map._grid[end_pos[1]][end_pos[0]] = end
        expected_set.add((end, "left_edge"))
        assert self.state.get_completed_components(start)

        result = self.state._traverse_connected_component(start, "top_edge")

        self.assertEqual(set(result), expected_set)

    def test_complex_city_traversal(self) -> None:
        pass


class TestPlaceMeeple(unittest.TestCase):
    pass
