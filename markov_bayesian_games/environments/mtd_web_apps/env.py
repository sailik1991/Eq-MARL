import numpy as np

__author__ = "Sailik Sengupta"


class Game(object):
    def __init__(self, start_states=None):
        """
        self.S = list               # State
        self.start_S = list         # Start States
        self.end_S = list           # Absorbing/Terminal States
        self.A = 3_D list           # [attacker/defender][type][state]
        """

        with open("./environments/mtd_web_apps/game.txt", "r") as f:
            num_states = int(f.readline())
            self.S = [i for i in range(num_states)]
            self.start_S = start_states if start_states is not None else self.S
            self.end_S = []

            defense_actions = [str(i) for i in self.S]
            A_D = [defense_actions for i in self.S]

            self.switching_cost = []
            for i in range(num_states):
                s_i = list(map(int, f.readline().split()))
                self.switching_cost.append(s_i)

            self.num_attacker_types = int(f.readline())
            self.attacker_type_probs = []
            A_A = []
            self.R_D = {}
            self.R_A = {}
            for theta in range(self.num_attacker_types):
                self.attacker_type_probs.append(float(f.readline()))
                num_attacks = int(f.readline())
                A_theta = list(map(str, f.readline().split("|")))

                for config in defense_actions:
                    rewards = f.readline().split(" ")
                    assert len(rewards) == num_attacks
                    for i in range(num_attacks):
                        k_i = "{}_{}_{}".format(theta, config, A_theta[i])
                        self.R_D[k_i], self.R_A[k_i] = list(map(float, rewards[i].split(",")))

                A_A.append([A_theta for i in self.S])

            self.A = [A_A, A_D]

        self.opt_pi = {
            0: {"pi_0": 0.5, "pi_1": 0.5, "pi_2": 0.0, "pi_3": 0.0},
            1: {"pi_0": 0.5, "pi_1": 0.5, "pi_2": 0.0, "pi_3": 0.0},
            2: {"pi_0": 0.5, "pi_1": 0.5, "pi_2": 0.0, "pi_3": 0.0},
            3: {"pi_0": 0.5, "pi_1": 0.5, "pi_2": 0.0, "pi_3": 0.0},
        }

    def get_start_state(self):
        return np.random.choice(self.start_S)

    def act(self, s, a_D, a_A, t=0):
        assert a_D in self.A[1][s], "Action {} not present in {}".format(
            a_D, self.A[1][s]
        )
        assert a_A in self.A[0][t][s], "Action {} not present in {}".format(
            a_A, self.A[0][t][s]
        )

        next_s = self.S[int(a_D)]
        k = "{}_{}_{}".format(t, a_D, a_A)
        return self.R_D[k] - 0.1 * self.switching_cost[s][next_s], self.R_A[k], next_s

    def is_end(self, s):
        assert s in self.S
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
