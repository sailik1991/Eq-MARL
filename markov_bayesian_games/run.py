from agents import StackelbergLearner, URSLearner, EXPLearner, StaticPolicyNoLearner
from environments.mtd_web_apps.env import Game

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import math

PLOT_OPT_DIST = False
PLOT_EPS = False
PLOT_R = True


def sample_act_from_policy(pi, epsilon=0.1):
    acts = list(pi.keys())
    if np.random.random() < epsilon:
        # random action selection
        probs = [1.0 / len(acts) for k in acts]
    else:
        # select action as per pi
        probs = [pi[k] for k in acts]

    acts = [a[a.startswith("pi_") and len("pi_") :] for a in acts]
    probs = [max(0.0, p) for p in probs]
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    return np.random.choice(acts, p=probs)


# Computes L2 distance between two policies
# policy format : {'pi_mon-LDAP': 0.5, 'pi_no-mon': 0.4}
def compute_policy_distance(p1, p2):
    distance = 0.0
    for k in p1.keys():
        distance += (p1[k] - p2[k]) ** 2
    return math.sqrt(distance)


def run(env, rl_agent, episodes=25, eps_d=0.1, optimal_policy=None):
    max_steps_per_episode = 25
    exploration_rate_decay = 0.9999

    # Initialize RL agent with Uniform Random Strategy
    rl_agent.initial_policy()
    eps_lengths = []
    distance_to_opt = {}
    state_rewards_eps = {}
    state_rewards_D = {}
    for i in range(episodes):
        if i % 50 == 0:
            print("Episode #{}".format(i))

        """
        Read discussion on decaying exploration rate.
        https://ai.stackexchange.com/questions/7923/learning-rate-decay-and-exploration-rate-decay
        As opposed to doing this after each action, we do this decay after every episode.
        A lower bound ensures that the agent isn't stuck with a bad policy just because LR -> 0
        Also, if agent is EXP-Q, no exploration is needed here (eps = 0). The agent policy accounts for it.
        """
        # The defender might have a static policy that does not learn. In such cases,
        # as per the stackelberg threat model, the attacker should be albe to select a best response.
        eps_d = max(eps_d * exploration_rate_decay, 0.05) if eps_d > 0.05 else eps_d
        eps_a = eps_d if eps_d > 0.0 else 0.2 * exploration_rate_decay
        exploration_rate_decay *= exploration_rate_decay

        j = 0
        s_t = env.get_start_state()
        while j < max_steps_per_episode:
            j += 1

            # Sample a policy to execute in state s_t
            pi_D, pi_A, theta = rl_agent.get_policy_in_state(s_t)
            a_D = sample_act_from_policy(pi_D, epsilon=eps_d)
            a_A = sample_act_from_policy(pi_A, epsilon=eps_a)

            # Save distance to optimal policy before acting in the env.
            if optimal_policy:
                diff_opt_pi = compute_policy_distance(optimal_policy[s_t], pi_D)
                try:
                    distance_to_opt[s_t].append(diff_opt_pi)
                except KeyError:
                    distance_to_opt[s_t] = [diff_opt_pi]

            # print('[DEBUG] Policy \nDef: {} \nAtt: {}'.format(pi_D, pi_A))
            # print('[DEBUG] State: {}, Def: {}, Att: {}'.format(s_t, a_D, a_A))

            # Act according to sampled policy
            r_D, r_A, s_next = env.act(s_t, a_D, a_A, theta)

            # Save rewards obtained in a state for plotting
            try:
                # state_rewards_D[s_t].append(r_D)
                state_rewards_eps[s_t].append(r_D)
            except KeyError:
                # state_rewards_D[s_t] = [r_D]
                state_rewards_eps[s_t] = [r_D]

            rl_agent.update_Q(s_t, a_D, a_A, r_D, r_A, s_next, theta)
            rl_agent.update_value_and_policy(s_t)

            s_t = s_next
            if env.is_end(s_t):
                break

        for state in state_rewards_eps.keys():
            avg_state = sum(state_rewards_eps[state]) / len(state_rewards_eps[state])
            try:
                state_rewards_D[state].append(avg_state)
            except KeyError:
                state_rewards_D[state] = [avg_state]
        state_rewards_eps = {}

        eps_lengths.append(j)

    # [Debug] Ensure that the Q-values are approx. equal to true reward values.
    # print("[DEBUG] {}'s learnt values:".format(rl_agent.get_name()))
    # rl_agent.print_Q()

    print("[DEBUG] {}'s policy:".format(rl_agent.get_name()))
    rl_agent.print_policy()

    return eps_lengths, state_rewards_D, distance_to_opt


def save_data(data, file_name="tmp.data"):
    with open("./outputs/{}.pickle".format(file_name), "wb") as f:
        pickle.dump(data, f)


def learn(env, learner=StackelbergLearner, num_try=2, eps_d=0.1):
    episode_lengths = []
    state_rewards_for_D = []
    distance_to_optimal_policy = []

    opt_pi = None
    # opt_pi = env.get_optimal_policy()

    for t in range(num_try):
        rl_agent = learner(env, discount_factor=0.8, alpha=0.05)
        el, srd, dto = run(
            env, rl_agent, episodes=200, optimal_policy=opt_pi, eps_d=eps_d
        )
        episode_lengths.append(el)
        state_rewards_for_D.append(srd)
        distance_to_optimal_policy.append(dto)

    return episode_lengths, state_rewards_for_D, distance_to_optimal_policy


def run_marl(env, learner=StackelbergLearner, eq="Nash"):
    try:
        # Use <no_such_file> to forcefully generate data file when one already exists.
        episode_lengths, state_rewards_for_D, distance_to_optimal_policy = pickle.load(
            # open("outputs/exp_data_{}<no_such_file>Learner.pickle".format(eq), "rb")
            open("outputs/exp_data_{}Learner.pickle".format(eq), "rb")
        )
    except:
        # For static policies, defender does not do exploitations
        # For EXP-Q learning, exploitation is part of the reward assessment.
        eps_d = 0.2 if eq == "SSE" else 0
        episode_lengths, state_rewards_for_D, distance_to_optimal_policy = learn(
            env, learner, num_try=6, eps_d=eps_d
        )
        save_data(
            (episode_lengths, state_rewards_for_D, distance_to_optimal_policy),
            file_name="exp_data_{}".format("{}Learner".format(eq)),
        )


if __name__ == "__main__":
    env = Game()

    """ Run MARL """
    run_marl(env, URSLearner, "URS")
    run_marl(env, EXPLearner, "EXP")
    run_marl(env, StackelbergLearner, "SSE")

    env_with_start_states = Game(start_states=[2, 3])
    run_marl(env_with_start_states, StaticPolicyNoLearner, "SPNL")
