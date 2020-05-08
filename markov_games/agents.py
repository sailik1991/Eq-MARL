import nashpy as nash
import numpy as np
import importlib
import pprint
import abc

pp = pprint.PrettyPrinter(indent=2)


class QLearner:
    def __init__(self, game_to_play, discount_factor, alpha, learning_rate_decay=0.98):
        __metaclass__ = abc.ABCMeta

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
            if self.GAME.is_end(s):
                self.V_D[s] = self.GAME.R[1][s][0][0]
                self.V_A[s] = self.GAME.R[0][s][0][0]

            # Initialize Q-value functions
            for d in self.A_D[s]:
                for a in self.A_A[s]:
                    sda = "{}_{}_{}".format(s, d, a)
                    if self.GAME.is_end(s):
                        self.Q_D[sda] = self.GAME.R[1][s][0][0]
                        self.Q_A[sda] = self.GAME.R[0][s][0][0]
                    else:
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

    @abc.abstractmethod
    def update_value_and_policy(self):
        raise NotImplementedError

    def print_Q(self):
        pp.pprint(self.Q_D)
        pp.pprint(self.Q_A)

    def print_policy(self):
        pp.pprint(self.policy_D)
        pp.pprint(self.policy_A)


class NashLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super(NashLearner, self).__init__(*args, **kwargs)
        self.name = 'NashLearner'

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
            # print('[DEBUG] Using Lemke Howson Enumeration.')
            D_s, A_s = list(g.lemke_howson_enumeration())[0]
        game_value = g[D_s, A_s]

        return game_value[0], game_value[1], D_s, A_s


class StackelbergLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super(StackelbergLearner, self).__init__(*args, **kwargs)
        self.name = 'SSELearner'
        self.lib = importlib.import_module('gurobi')

    def get_name(self):
        return self.name

    # Given the Q-values, update (1) the value of state s and (2) the policy of the agents
    def update_value_and_policy(self, s):
        num_d = len(self.A_D[s])
        num_a = len(self.A_A[s])

        m = self.lib.Model('MIQP')
        m.setParam('OutputFlag', 0)
        m.setParam('LogFile', '')
        m.setParam('LogToConsole', 0)

        # Add defender stategies to the model
        x = []
        for i in range(num_d):
            n = 'x_{}'.format(self.A_D[s][i])
            x.append(m.addVar(lb=0, ub=1, vtype=self.lib.GRB.CONTINUOUS, name=n))
        m.update()

        # Add defender stategy constraints
        con = self.lib.LinExpr()
        for i in range(num_d):
            con.add(x[i])
        m.addConstr(con == 1)

        obj = self.lib.QuadExpr()
        M = 100000000

        q = []
        for i in range(num_a):
            n = 'q_{}'.format(self.A_A[s][i])
            q.append(m.addVar(lb=0, ub=1, vtype=self.lib.GRB.INTEGER, name=n))

        V_a = m.addVar(lb=-self.lib.GRB.INFINITY, ub=self.lib.GRB.INFINITY, vtype=self.lib.GRB.CONTINUOUS, name="V_a")
        m.update()

        # Get Defender/Leader's Q-value matrix
        M_D = []
        M_A = []
        for d in range(num_d):
            row_D = []
            row_A = []
            for a in range(num_a):
                k = '{}_{}_{}'.format(s, self.A_D[s][d], self.A_A[s][a])
                row_D.append(self.Q_D[k])
                row_A.append(self.Q_A[k])
            M_D.append(row_D)
            M_A.append(row_A)

        # Update objective function
        for i in range(num_d):
            for j in range(num_a):
                obj.add(M_D[i][j] * x[i] * q[j])

        # Add constraints to make attaker have a pure strategy
        con = self.lib.LinExpr()
        for j in range(num_a):
            con.add(q[j])
        m.addConstr(con == 1)

        # Add constrains to make attacker select dominant pure strategy
        for j in range(num_a):
            val = self.lib.LinExpr()
            val.add(V_a)
            for i in range(num_d):
                val.add(float(M_A[i][j]) * x[i], -1.0)
            m.addConstr(val >= 0, q[j].getAttr('VarName') + "_lb")
            m.addConstr(val <= (1 - q[j]) * M, q[j].getAttr('VarName') + "_ub")

        # Set objective funcion as all attackers have now been considered
        m.setObjective(obj, self.lib.GRB.MAXIMIZE)

        # Solve MIQP
        m.optimize()

        self.V_D[s] = float(m.ObjVal)
        for var in m.getVars():
            if 'x_' in var.varName:
                self.policy_D[s][var.varName.replace('x_', 'pi_')] = float(var.x)
            if 'q_' in var.varName:
                self.policy_A[s][var.varName.replace('q_', 'pi_')] = float(var.x)
            if 'V_a' in var.varName:
                self.V_A[s] = float(var.x)


class URSLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super(URSLearner, self).__init__(*args, **kwargs)
        self.name = 'URSLearner'
        self.lib = importlib.import_module('gurobi')

    def get_name(self):
        return self.name

    # Given the Q-values, update (1) the value of state s and (2) the policy of the agents
    def update_value_and_policy(self, s):
        num_d = len(self.A_D[s])
        num_a = len(self.A_A[s])

        m = self.lib.Model('MIQP')
        m.setParam('OutputFlag', 0)
        m.setParam('LogFile', '')
        m.setParam('LogToConsole', 0)

        x_ur = 1 / num_d
        obj = self.lib.LinExpr()
        M = 100000000
        q = []
        for i in range(num_a):
            n = 'q_{}'.format(self.A_A[s][i])
            q.append(m.addVar(lb=0, ub=1, vtype=self.lib.GRB.INTEGER, name=n))

        V_a = m.addVar(lb=-self.lib.GRB.INFINITY, ub=self.lib.GRB.INFINITY, vtype=self.lib.GRB.CONTINUOUS, name="V_a")
        m.update()

        # Get Defender/Leader's Q-value matrix
        M_D = []
        M_A = []
        for d in range(num_d):
            row_D = []
            row_A = []
            for a in range(num_a):
                k = '{}_{}_{}'.format(s, self.A_D[s][d], self.A_A[s][a])
                row_D.append(self.Q_D[k])
                row_A.append(self.Q_A[k])
            M_D.append(row_D)
            M_A.append(row_A)

        # Update objective function
        for i in range(num_d):
            for j in range(num_a):
                obj.add(M_D[i][j] * x_ur * q[j])

        # Add constraints to make attaker have a pure strategy
        con = self.lib.LinExpr()
        for j in range(num_a):
            con.add(q[j])
        m.addConstr(con == 1)

        # Add constrains to make attacker select dominant pure strategy
        for j in range(num_a):
            val = self.lib.LinExpr()
            val.add(V_a)
            for i in range(num_d):
                val.add(float(M_A[i][j]) * x_ur, -1.0)
            m.addConstr(val >= 0, q[j].getAttr('VarName') + "_lb")
            m.addConstr(val <= (1 - q[j]) * M, q[j].getAttr('VarName') + "_ub")

        # Set objective funcion as all attackers have now been considered
        m.setObjective(obj, self.lib.GRB.MAXIMIZE)

        # Solve MIQP
        m.optimize()

        self.V_D[s] = float(m.ObjVal)
        for var in m.getVars():
            for a_D in self.A_D[s]:
                self.policy_D[s]["pi_{}".format(a_D)] = x_ur
            if 'q_' in var.varName:
                self.policy_A[s][var.varName.replace('q_', 'pi_')] = float(var.x)
            if 'V_a' in var.varName:
                self.V_A[s] = float(var.x)


class EXPLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super(EXPLearner, self).__init__(*args, **kwargs)
        self.name = "EXPLearner"
        self.lib = importlib.import_module("gurobi")
        self.sum_R = {}
        for s in self.S:
            for a in self.A_D[s]:
                self.sum_R["{}_{}".format(s, a)] = 0.0

    def get_name(self):
        return self.name

    def get_defender_startegy(self, s):
        self.new_policy = {}
        for i in range(len(self.A_D[s])):
            Q_t = 0.0
            for a_A in self.A_A[s]:
                sda = "{}_{}_{}".format(s, self.A_D[s][i], a_A)
                Q_t += self.Q_D[sda]
            self.sum_R["{}_{}".format(s, self.A_D[s][i])] += Q_t / self.policy_D[s]["pi_{}".format(self.A_D[s][i])]

        x = []
        for i in range(len(self.A_D[s])):
            denominator = 0.001
            for j in range(len(self.A_D[s])):
                denominator += np.exp(
                    0.1
                    / len(self.A_D[s])
                    * (
                        self.sum_R["{}_{}".format(s, self.A_D[s][j])]
                        - self.sum_R["{}_{}".format(s, self.A_D[s][i])]
                    )
                )
            x.append(0.9 / denominator + 0.1 / len(self.A_D[s]))

        x_L1 = np.nansum(x)
        x = [i / x_L1 for i in x]
        return x

    # Given the Q-values, update (1) the value of state s and (2) the policy of the agents
    def update_value_and_policy(self, s):
        x = self.get_defender_startegy(s)
        num_d = len(self.A_D[s])
        num_a = len(self.A_A[s])

        m = self.lib.Model('MIQP')
        m.setParam('OutputFlag', 0)
        m.setParam('LogFile', '')
        m.setParam('LogToConsole', 0)

        obj = self.lib.LinExpr()
        M = 100000000
        q = []
        for i in range(num_a):
            n = 'q_{}'.format(self.A_A[s][i])
            q.append(m.addVar(lb=0, ub=1, vtype=self.lib.GRB.INTEGER, name=n))

        V_a = m.addVar(lb=-self.lib.GRB.INFINITY, ub=self.lib.GRB.INFINITY, vtype=self.lib.GRB.CONTINUOUS, name="V_a")
        m.update()

        # Get Defender/Leader's Q-value matrix
        M_D = []
        M_A = []
        for d in range(num_d):
            row_D = []
            row_A = []
            for a in range(num_a):
                k = '{}_{}_{}'.format(s, self.A_D[s][d], self.A_A[s][a])
                row_D.append(self.Q_D[k])
                row_A.append(self.Q_A[k])
            M_D.append(row_D)
            M_A.append(row_A)

        # Update objective function
        for i in range(num_d):
            for j in range(num_a):
                obj.add(M_D[i][j] * x[i] * q[j])

        # Add constraints to make attaker have a pure strategy
        con = self.lib.LinExpr()
        for j in range(num_a):
            con.add(q[j])
        m.addConstr(con == 1)

        # Add constrains to make attacker select dominant pure strategy
        for j in range(num_a):
            val = self.lib.LinExpr()
            val.add(V_a)
            for i in range(num_d):
                val.add(float(M_A[i][j]) * x[i], -1.0)
            m.addConstr(val >= 0, q[j].getAttr('VarName') + "_lb")
            m.addConstr(val <= (1 - q[j]) * M, q[j].getAttr('VarName') + "_ub")

        # Set objective funcion as all attackers have now been considered
        m.setObjective(obj, self.lib.GRB.MAXIMIZE)

        # Solve MIQP
        m.optimize()

        self.V_D[s] = float(m.ObjVal)
        for var in m.getVars():
            for i in range(len(self.A_D[s])):
                self.policy_D[s]["pi_{}".format(self.A_D[s][i])] = x[i]
            if 'q_' in var.varName:
                self.policy_A[s][var.varName.replace('q_', 'pi_')] = float(var.x)
            if 'V_a' in var.varName:
                self.V_A[s] = float(var.x)
        del m
