from agents import NashLearner, StackelbergLearner
from environments.ids_place_game import Game

# from environments.mtd_switching_costs import Game

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import math

PLOT_OPT_DIST = True
PLOT_EPS = False
PLOT_R = False


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


def run(env, rl_agent, episodes=25, optimal_policy=None):
    max_steps_per_episode = 100
    exploration_rate_decay = 0.9999

    # Initialize RL agent with Uniform Random Strategy
    rl_agent.initial_policy()
    eps_lengths = []
    distance_to_opt = {}
    state_rewards_eps = {}
    state_rewards_D = {}
    for i in range(episodes):

        # Read discussion on decaying exploration rate.
        # https://ai.stackexchange.com/questions/7923/learning-rate-decay-and-exploration-rate-decay
        # As opposed to doing this after each action, we do this decay after every episode.
        # A lower bound ensures that the agent isn't stuck with a bad policy just because LR -> 0
        epsilon = max(0.1 * exploration_rate_decay, 0.05)
        exploration_rate_decay *= exploration_rate_decay
        j = 0
        s_t = env.get_start_state()
        while j < max_steps_per_episode:
            j += 1

            # Sample a policy to execute in state s_t
            pi_D, pi_A = rl_agent.get_policy_in_state(s_t)
            a_D = sample_act_from_policy(pi_D, epsilon=epsilon)
            a_A = sample_act_from_policy(pi_A, epsilon=epsilon)

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
            r_D, r_A, s_next = env.act(s_t, a_D, a_A)

            # Save rewards obtained in a state for plotting
            try:
                # state_rewards_D[s_t].append(r_D)
                state_rewards_eps[s_t].append(r_D)
            except KeyError:
                # state_rewards_D[s_t] = [r_D]
                state_rewards_eps[s_t] = [r_D]

            rl_agent.update_Q(s_t, a_D, a_A, r_D, r_A, s_next)
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
    print("[DEBUG] {}'s learnt values:".format(rl_agent.get_name()))
    rl_agent.print_Q()

    print("[DEBUG] {}'s policy:".format(rl_agent.get_name()))
    rl_agent.print_policy()

    return eps_lengths, state_rewards_D, distance_to_opt


def group_episode_lengths(lengths):
    i = 0
    group_size = 10
    grouped_lengths = []

    while (i + 1) * group_size <= len(lengths):
        grouped_lengths.append(
            sum([j for j in lengths[i * group_size : (i + 1) * group_size]])
            / group_size
        )
        i += 1

    return grouped_lengths


def plot_scalar(y):
    y = np.array(y)
    rewards_avg = np.average(y, axis=0)
    plt.plot([i for i in range(len(rewards_avg))], rewards_avg, label="avg", color="C0")


def plot_state_scalars(state_rewards_D, axl, file_suffix="NashLearner"):
    state_rewards_D = state_rewards_D[0]
    states = state_rewards_D.keys()

    for s in states:
        axl[s].set_title("State {}".format(s))
        axl[s].plot(
            [i for i in range(len(state_rewards_D[s]))],
            state_rewards_D[s],
            label=file_suffix,
        )
        axl[s].legend(loc="upper right")


def save_data(data, file_name="tmp.data"):
    with open("./outputs/{}.pickle".format(file_name), "wb") as f:
        pickle.dump(data, f)


def learn(env, learner=NashLearner, num_try=1):
    episode_lengths = []
    state_rewards_for_D = []
    distance_to_optimal_policy = []

    opt_pi = None
    # opt_pi = env.get_optimal_policy()

    for t in range(num_try):
        # rl_agent = learner(env, discount_factor=0.6, alpha=0.05)
        rl_agent = learner(env, discount_factor=0.9, alpha=0.05)
        el, srd, dto = run(env, rl_agent, episodes=99, optimal_policy=opt_pi)
        episode_lengths.append(group_episode_lengths(el))
        state_rewards_for_D.append(srd)
        distance_to_optimal_policy.append(dto)

    return episode_lengths, state_rewards_for_D, distance_to_optimal_policy


def run_marl(env, axl, learner=NashLearner, eq="Nash"):
    try:
        episode_lengths, state_rewards_for_D, distance_to_optimal_policy = pickle.load(
            open("outputs/exp_data_{}Learner.pickle".format(eq), "rb")
        )
    except:
        episode_lengths, state_rewards_for_D, distance_to_optimal_policy = learn(
            env, learner, num_try=1
        )
        save_data(
            (episode_lengths, state_rewards_for_D, distance_to_optimal_policy),
            file_name="exp_data_{}".format("{}Learner".format(eq)),
        )

    if PLOT_OPT_DIST:
        plot_state_scalars(distance_to_optimal_policy, axl, file_suffix=eq)
        fig.text(0.02, 0.5, "L2 distance to pi* --> ", va="center", rotation="vertical")
        fig.text(0.5, 0.02, "episodes * steps -->", ha="center")
        plt.savefig("./images/state_pi_diff.png")
    if PLOT_R:
        plot_state_scalars(state_rewards_for_D, axl, file_suffix=eq)
        fig.text(0.015, 0.5, "R (defender) -->", va="center", rotation="vertical")
        fig.text(0.5, 0.04, "episodes -->", ha="center")
        plt.savefig("./images/state_rewards.png")
    if PLOT_EPS:
        plot_scalar(episode_lengths)
        plt.xlabel("episodes -->")
        plt.ylabel("steps --> ")
        plt.savefig("./images/episode_length.png")


if __name__ == "__main__":
    env = Game()
    sns.set()
    # sns.set_context("paper")
    sns.set_context("talk")
    sns.set_palette("deep")
    fig, axl = None, None
    if PLOT_OPT_DIST or PLOT_R:
        fig, axl = plt.subplots(
            len(env.S), sharex=True, figsize=(7, 8.5), gridspec_kw={"hspace": 0.4}
        )
    run_marl(env, axl, NashLearner, 'Nash')
    run_marl(env, axl, StackelbergLearner, 'SSE')

    # try:
    #     episode_lengths, state_rewards_for_D, distance_to_optimal_policy = pickle.load(
    #         open("outputs/exp_data_SSELearner.pickle", "rb")
    #     )
    # except:
    #     episode_lengths, state_rewards_for_D, distance_to_optimal_policy = learn(
    #         env, StackelbergLearner, num_try=1
    #     )
    #     save_data(
    #         (episode_lengths, state_rewards_for_D, distance_to_optimal_policy),
    #         file_name="exp_data_{}".format('StackelbergLearner'),
    #     )
    # if plot_opt_policy_distance:
    #     plot_state_scalars(distance_to_optimal_policy, axl, file_suffix="SSE")
    # if plot_rewards:
    #     plot_state_scalars(state_rewards_for_D, axl, file_suffix="SSE")
    # if plot_eps_lengths:
    #     plot_scalar(episode_lengths)

    # plot_state_scalars(state_rewards_for_D, axl, file_suffix="SSE")
    # plot_state_scalars(distance_to_optimal_policy, axl, file_suffix='SSE')

    """
    Code to plot episode length data into a file.
    """
    # plt.xlabel('episodes -->')
    # plt.ylabel('steps --> ')
    # plt.savefig("./images/episode_length.png")

    """
    Code to plot state based scalar values into a file.
    """
    # fig.text(0.5, 0.04, 'episodes * steps -->', ha='center')
    # fig.text(0.015, 0.5, 'L2 distance to optimal policy --> ', va='center', rotation='vertical')
    # plt.savefig("./images/state_rewards.png")

    # plot_episode_lengths(episode_lengths, file_suffix=rl_agent.get_name())

    # rl_agent = StackelbergLearner(env, discount_factor=0.5, alpha=0.1)
    # run(env, rl_agent, episodes=50)

    # plt.savefig("./defender_rewards_mtd.png")
