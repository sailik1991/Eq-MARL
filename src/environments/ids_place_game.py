import numpy as np

__author__ = "Sailik Sengupta"


class Game(object):
    def __init__(self):
        self.S = [0, 1, 2, 3]
        self.start_S = [1, 2, 3]
        self.end_S = [0]

        self.A = [
            # Row player (Attacker) actions corresponding to each state
            [
                ["success"],
                ["no-op", "exp-LDAP"],
                ["no-op", "exp-Web", "exp-FTP"],
                ["no-op", "exp-FTP"]
            ],
            # Column player (Defender) actions corresponding to each state
            [
                ["lost"],
                ["no-mon", "mon-LDAP"],
                ["no-mon", "mon-Web", "mon-FTP"],
                ["no-mon", "mon-FTP"]
            ],
        ]

        # R(s, a1, a2)
        self.R = [
            [
                [[25]],
                [[0, 0], [5, -5]],
                [[0, 0, 0], [7, -6, 9], [10, 10, -8]],
                [[0, 0], [10, -10]],
            ],
            [
                [[-25]],
                [[0, -3], [-5, 5]],
                [[0, -2, -3], [-7, 5, -9], [-10, -10, 7]],
                [[0, -2], [-10, 8]],
            ],
        ]

        # T(s, a1, a2, s')
        self.T = [
            [[(1, 0, 0, 0)]],
            [
                [(0, 1, 0, 0), (0, 1, 0, 0)],
                [(0, 0.5, 0.5, 0), (0, 0.9, 0.1, 0)]
            ],
            [
                [(0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)],
                [(0, 0, 0.3, 0.7), (0, 0.3, 0.1, 0.6), (0, 0, 0.4, 0.6)],
                [(0, 0, 0.4, 0.6), (0, 0, 0.3, 0.7), (0, 0.5, 0.1, 0.4)],
            ],
            [
                [(0, 0, 0, 1), (0, 0, 0, 1)],
                [(0.8, 0, 0, 0.2), (0.4, 0.4, 0, 0.2)]
            ],
        ]

    def get_start_state(self):
        return np.random.choice(self.start_S)

    def act(self, s, a1, a2):
        try:
            a1 = self.A[1][s].index(a1)
            a2 = self.A[0][s].index(a2)
        except ValueError:
            print(a1, self.A[1][s])
            print(a2, self.A[0][s])
            raise ValueError('[ERROR] Action names are not present in the game!')

        next_s = np.random.choice(self.S, p=self.T[s][a2][a1])
        reward_D = self.R[1][s][a2][a1]
        reward_A = self.R[0][s][a2][a1]
        return reward_D, reward_A, next_s

    def is_end(self, s):
        if s in self.end_S:
            return True
        return False

    def get_states(self):
        return self.S

    def get_actions(self):
        return self.A
