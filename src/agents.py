class Q_learner:
    def __init__(self, game_to_play, discount_factor, alpha):
        self.GAME = game_to_play
        self.DISCOUNT_FACTOR = discount_factor
        self.ALPHA = alpha

        self.S = self.GAME.get_states()
        self.A_A, self.A_D = self.GAME.get_actions()

        # Define value and Q-value functions (for two players)
        self.V_D = {}
        self.V_A = {}
        self.Q_D = {}
        self.Q_A = {}

        for s in self.S():
            # Initialize value functions
            self.V_D[s] = self.V_A[s] = 0

            # Initialize Q-value functions
            for d in self.A_D:
                for a in self.A_A:
                    sda = "{}_{}_{}".format(s, d, a)
                    self.Q_D[sda] = 0
                    self.Q_A[sda] = 0

    # Q-learning update for two-player setting:
    # Q_D(s, d, a) = (1 - alpha) * Q_D(s, d, a) + alpha * (R_D + gamma * V_D)
    def update_Q(self, s_t, d_t, a_t, r, s_next):
        assert s_t in self.S and s_next in self.S
        assert d_t in self.A_D[s_t] and a_t in self.A_A[s_t]

        self.Q_D[s_t, d_t, a_t] = (1 - self.alpha) * self.Q_D[s_t, d_t, a_t] +\
                                  self.alpha * (r + self.DISCOUNT_FACTOR * self.V_D[s_next])

    def get_values(self):
        raise NotImplementedError