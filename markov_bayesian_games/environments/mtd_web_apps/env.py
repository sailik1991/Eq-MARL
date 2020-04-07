import numpy as np

__author__ = "Sailik Sengupta"

class Game(object):
    def __init__(self):
        '''
        self.S = list               # State
        self.start_S = list         # Start States
        self.end_S = list           # Absorbing/Terminal States
        self.A = 3_D list           # [attacker/defender][type][state]
        '''

        with open('./environments/mtd_web_apps/game.txt', 'r') as f:
            num_states = int(f.readline())
            self.S = [i for i in range(num_states)]
            self.start_S = self.S
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
                A_theta = list(map(str, f.readline().split('|')))

                for config in defense_actions:
                    rewards = f.readline().split(' ')
                    assert len(rewards) == num_attacks
                    for i in range(num_attacks):
                        k_i = '{}_{}_{}'.format(theta, config, A_theta[i])
                        self.R_D[k_i], self.R_A[k_i] = tuple(map(float, rewards[i].split((','))))

                A_A.append([A_theta for i in self.S])

            self.A = [A_A, A_D]
            print(A_D)

        # Player types
        # self.attacker_type_probs = [0.4, 0.5, 0.1]
        # self.num_attacker_types = len(self.attacker_type_probs)
        #
        # attack_actions = ["no-op", "low-0", "med-0", "low-1", "high-1", "med-2", "high-2", "high-3"]
        # defense_actions = [str(i) for i in self.S]
        # self.A = [
        #     # Attacker actions
        #     [
        #         # Type 1 - Script Kiddie
        #         [[a for a in attack_actions if 'low' in a or 'no' in a] for i in self.S],
        #         # Type 2 - SQL Database Hacker
        #         [[a for a in attack_actions if '1' in a or '2' in a or 'no' in a] for i in self.S],
        #         # Type 3 - Nation State
        #         [attack_actions for i in self.S]
        #     ],
        #     # Defender's actions
        #     [defense_actions for i in self.S]
        # ]

        self.opt_pi = {
            0: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0},
            1: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0},
            2: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0},
            3: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0}
        }

    def get_start_state(self):
        return np.random.choice(self.start_S)

    def act(self, s, a_D, a_A, t=0):
        assert a_D in self.A[1][s], "Action {} not present in {}".format(a_D, self.A[1][s])
        assert a_A in self.A[0][t][s], "Action {} not present in {}".format(a_A, self.A[0][t][s])

        next_s = self.S[int(a_D)]
        k = '{}_{}_{}'.format(t, a_D, a_A)
        return self.R_D[k], self.R_A[k], next_s

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
