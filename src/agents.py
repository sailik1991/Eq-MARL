import nashpy as nash
import pprint

pp = pprint.PrettyPrinter(indent=2)


class QLearner:
    def __init__(self, game_to_play, discount_factor, alpha):
        self.GAME = game_to_play
        self.DISCOUNT_FACTOR = discount_factor
        self.ALPHA = alpha

        self.S = self.GAME.get_states()
        # 2-D array indexed by [state][action]
        self.A_A, self.A_D = self.GAME.get_actions()

        # Define policy in each state (for two players) - indexed by state
        self.policy_D = {}
        self.policy_A = {}
        for s in self.S:
            self.policy_D[s] = {}
            self.policy_A[s] = {}

        # Define value and Q-value functions (for two players)
        # Indexed by states in S
        self.V_D = {}
        self.V_A = {}
        # Indexed by the string "state_P1-act_P2-act"
        self.Q_D = {}
        self.Q_A = {}

        for s in self.S:
            # Initialize value functions
            self.V_D[s] = self.V_A[s] = 0

            # Initialize Q-value functions
            for d in self.A_D[s]:
                for a in self.A_A[s]:
                    sda = "{}_{}_{}".format(s, d, a)
                    self.Q_D[sda] = 0
                    self.Q_A[sda] = 0

    # Q-learning update for two-player setting:
    # Q_D(s, d, a) = (1 - alpha) * Q_D(s, d, a) + alpha * (R_D + gamma * V_D)
    def update_Q(self, s_t, d_t, a_t, r_D, r_A, s_next):
        assert s_t in self.S and s_next in self.S
        assert d_t in self.A_D[s_t] and a_t in self.A_A[s_t]

        sda_t = "{}_{}_{}".format(s_t, d_t, a_t)

        self.Q_D[sda_t] = (1 - self.ALPHA) * self.Q_D[sda_t] + self.ALPHA * (
            r_D + self.DISCOUNT_FACTOR * self.V_D[s_next]
        )

        self.Q_A[sda_t] = (1 - self.ALPHA) * self.Q_A[sda_t] + self.ALPHA * (
            r_A + self.DISCOUNT_FACTOR * self.V_A[s_next]
        )

    def print_Q(self):
        pp.pprint(self.Q_A)
        pp.pprint(self.Q_D)

    # Initialize with a uniform random policy over all actions in the state
    def initial_policy(self):
        for s in self.S:
            for i in range(len(self.A_D[s])):
                self.policy_D[s]["pi_{}".format(self.A_D[s][i])] = 1.0 / len(
                    self.A_D[s]
                )
            for i in range(len(self.A_A[s])):
                self.policy_A[s]["pi_{}".format(self.A_A[s][i])] = 1.0 / len(
                    self.A_A[s]
                )

    def get_policy_in_state(self, s):
        assert s in self.S
        return self.policy_D[s], self.policy_A[s]

    def get_values(self):
        raise NotImplementedError


class NashLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super(NashLearner, self).__init__(*args, **kwargs)

    def get_name(self):
        return self.name

    # Given the Q-values, update (1) the value of state s and (2) the policy of the agents
    def update_value_and_policy(self, s):
        num_d = len(self.A_D[s])
        num_a = len(self.A_A[s])

        # Compute the Q-value matrix of the two players
        M_D = []
        M_A = []
        for d in range(num_d):
            row_D = []
            row_A = []
            for a in range(num_a):
                k = "{}_{}_{}".format(s, self.A_D[s][d], self.A_A[s][a])
                row_D.append(self.Q_D[k])
                row_A.append(self.Q_A[k])
            M_D.append(row_D)
            M_A.append(row_A)

        self.V_D[s], self.V_A[s], strategy_D, strategy_A = self.find_nash(M_D, M_A)

        assert len(self.A_D[s]) == len(strategy_D)
        assert len(self.A_A[s]) == len(strategy_A)

        for i in range(len(self.A_D[s])):
            action_name = self.A_D[s][i]
            self.policy_D[s]["pi_{}".format(action_name)] = strategy_D[i]
        for i in range(len(self.A_A[s])):
            action_name = self.A_A[s][i]
            self.policy_A[s]["pi_{}".format(action_name)] = strategy_A[i]

    @staticmethod
    def find_nash(R_1, R_2):
        g = nash.Game(R_1, R_2)
        try:
            D_s, A_s = list(g.support_enumeration())[0]
        except IndexError:
            # When there exists no pure strategy nash eq. Unfortunately, this might not always
            # help when the game is degenerate: https://github.com/drvinceknight/Nashpy/issues/35
            D_s, A_s = list(g.lemke_howson_enumeration())[0]
        game_value = g[D_s, A_s]

        return game_value[0], game_value[1], D_s, A_s
