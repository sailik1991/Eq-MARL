import numpy as np

__author__ = "Sailik Sengupta"

class Game(object):
    def __init__(self):
        self.S = [0, 1, 2, 3]
        self.start_S = [0, 1, 2, 3]
        self.end_S = []
        self.switching_cost = [
            [0, 2, 6, 10],
            [2, 0, 9, 5],
            [6, 9, 0, 2],
            [10, 5, 2, 0]
        ]

        attack_actions = ["no-op", "low-0", "med-0", "low-1", "high-1", "med-2", "high-2", "high-3"]
        defense_actions = [str(i) for i in self.S]
        self.A = [
            # Attacker actions
            [attack_actions for i in self.S],
            # Defender's actions
            [defense_actions for i in self.S]
        ]

        self.opt_pi = {
            0: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0},
            1: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0},
            2: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0},
            3: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0}
        }

    def get_start_state(self):
        return np.random.choice(self.start_S)

    def act(self, s, a1, a2):
        assert a1 in self.A[1][s]
        assert a2 in self.A[0][s]

        next_s = self.S[int(a1)]
        return self.get_reward_D(s, a1, a2), self.get_reward_A(a1, a2), next_s

    def get_reward_D(self, s, a1, a2):
        r = 0 if s == int(a1) else -1 * self.switching_cost[s][int(a1)]
        if a1 in a2:
            if 'low' in a2:
                return r - 3.33
            if 'med' in a2:
                return r - 6.67
            if 'high' in a2:
                return r - 10.0
            raise ValueError('Attack \'{}\' has invalid name!'.format(a2))
        else:
            return r

    @staticmethod
    def get_reward_A(a1, a2):
        if a1 in a2:
            if 'low' in a2:
                return 3.33
            if 'med' in a2:
                return 6.67
            if 'high' in a2:
                return 10.0
            raise ValueError('Attack \'{}\' has invalid name!'.format(a2))
        else:
            return 0.0

    def is_end(self, s):
        if s in self.end_S:
            return True
        return False

    def get_states(self):
        return self.S

    def get_actions(self):
        return self.A

    # This should only be used for comparison
    # Computed using the true T and R matrices.
    def get_optimal_policy(self):
        return self.opt_pi
