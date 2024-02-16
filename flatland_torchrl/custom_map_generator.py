import numpy as np

from flatland.core.transition_map import GridTransitionMap
from flatland.core.grid.rail_env_grid import RailEnvTransitions

transitions = RailEnvTransitions()
cells = transitions.transition_list

empty = cells[0]

vertical_straight = cells[1]
horizontal_straight = transitions.rotate_transition(vertical_straight, 90)

left_switch_from_south = cells[2]
left_switch_from_west = transitions.rotate_transition(left_switch_from_south, 90)
left_switch_from_north = transitions.rotate_transition(left_switch_from_south, 180)
left_switch_from_east = transitions.rotate_transition(left_switch_from_south, 270)

diamond_crossing = cells[3]

left_slip_from_south = cells[4]
left_slip_from_west = transitions.rotate_transition(left_slip_from_south, 90)
left_slip_from_north = transitions.rotate_transition(left_slip_from_south, 180)
left_slip_from_east = transitions.rotate_transition(left_slip_from_south, 270)

right_double_slip_vertical = cells[5]
right_double_slip_horizontal = transitions.rotate_transition(
    right_double_slip_vertical, 90
)

symmetrical_slip_from_south = cells[6]
symmetrical_slip_from_west = transitions.rotate_transition(
    symmetrical_slip_from_south, 90
)
symmetrical_slip_from_north = transitions.rotate_transition(
    symmetrical_slip_from_south, 180
)
symmetrical_slip_from_east = transitions.rotate_transition(
    symmetrical_slip_from_south, 270
)

dead_end_from_south = cells[7]
dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)

right_turn_from_south = cells[8]
right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
right_turn_from_east = transitions.rotate_transition(right_turn_from_south, 270)

left_turn_from_south = cells[9]
left_turn_from_west = transitions.rotate_transition(left_turn_from_south, 90)
left_turn_from_north = transitions.rotate_transition(left_turn_from_south, 180)
left_turn_from_east = transitions.rotate_transition(left_turn_from_south, 270)

right_switch_from_south = cells[10]
right_switch_from_west = transitions.rotate_transition(right_switch_from_south, 90)
right_switch_from_north = transitions.rotate_transition(right_switch_from_south, 180)
right_switch_from_east = transitions.rotate_transition(right_switch_from_south, 270)


def generate_custom_rail(map_name: str):
    if map_name == "figure_eight":
        print("Using figure of eight map.")
        rail_map = np.array(
            [
                [empty]
                + [right_turn_from_south]
                + [horizontal_straight]
                + [right_turn_from_west]
                + [empty]
            ]
            + [
                [left_turn_from_east]
                + [right_switch_from_east]
                + [horizontal_straight]
                + [left_switch_from_west]
                + [right_turn_from_west]
            ]
            + [
                [right_turn_from_east]
                + [horizontal_straight] * (3)
                + [left_turn_from_west]
            ],
            dtype=np.uint16,
        )
        train_stations = [
            [((0, 2), 0)],
            [((2, 2), 0)],
        ]

        city_positions = [(0, 2), (2, 2)]
        city_orientations = [1, 1]

        agents_hints = {
            "city_positions": city_positions,
            "train_stations": train_stations,
            "city_orientations": city_orientations,
        }

        optionals = {"agents_hints": agents_hints}

        rail = GridTransitionMap(
            width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions
        )
        rail.grid = rail_map

        return rail, optionals

    if map_name == "line_map":
        print("Using line map.")
        map_width = 10
        map_height = 5
        rail_map = np.array(
            [[empty] * map_width]
            + [
                [right_turn_from_south]
                + [horizontal_straight] * (map_width - 2)
                + [right_turn_from_west]
            ]
            + [
                [right_turn_from_east]
                + [horizontal_straight] * (map_width - 2)
                + [left_turn_from_west]
            ]
            + [[empty] * map_width]
            + [[empty] * map_width],
            dtype=np.uint16,
        )

        train_stations = [
            [((2, map_width - 2), 0)],
            [((2, 0), 0)],
        ]

        city_positions = [(2, map_width - 2), (2, 0)]
        city_orientations = [1, 1]

        agents_hints = {
            "city_positions": city_positions,
            "train_stations": train_stations,
            "city_orientations": city_orientations,
        }

        optionals = {"agents_hints": agents_hints}

        rail = GridTransitionMap(
            width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions
        )
        rail.grid = rail_map

        return rail, optionals

    print("Map name not recognized.")
