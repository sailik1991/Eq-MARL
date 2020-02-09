from agents import NashLearner, StackelbergLearner
# from environments.ids_place_game import Game

from environments.mtd_switching_costs import Game

import matplotlib.pyplot as plt
import numpy as np
import pickle
import math


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
    state_rewards_D = {}
    for i in range(episodes):

        # Read discussion on decaying exploration rate.
        # https://ai.stackexchange.com/questions/7923/learning-rate-decay-and-exploration-rate-decay
        # As opposed to doing this after each action, we do this decay after every episode.
        # A lower bound ensure that if the agent isn't stuck with a back policy just because LR -> 0
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
                state_rewards_D[s_t].append(r_D)
            except KeyError:
                state_rewards_D[s_t] = []

            rl_agent.update_Q(s_t, a_D, a_A, r_D, r_A, s_next)
            rl_agent.update_value_and_policy(s_t)

            if env.is_end(s_t):
                break

            s_t = s_next

        eps_lengths.append(j)

    # [Debug] Ensure that the Q-values are approx. equal to true reward values.
    print("[DEBUG] {}'s policy:".format(rl_agent.get_name()))
    rl_agent.print_Q()
    rl_agent.print_policy()

    return eps_lengths, state_rewards_D, distance_to_opt


def group_episode_rewards(rewards):
    i = 0
    group_size = 10
    group_rewards = []

    while (i + 1) * group_size <= len(rewards):
        group_rewards.append(
            sum([j for j in rewards[i * group_size : (i + 1) * group_size]])
            / group_size
        )
        i += 1

    return group_rewards


def plot_scalar(y, file_suffix="NashLearner"):
    y = np.array(y)
    rewards_avg = np.average(y, axis=0)
    rewards_var = np.var(y, axis=0)

    plt.plot([i for i in range(len(rewards_avg))], rewards_avg, label="avg", color="C0")
    # plt.plot([i for i in range(len(rewards_avg))], [rewards_avg[i] - rewards_var[i] for i in range(len(rewards_avg))], label='avg - var', color='C1')
    # plt.plot([i for i in range(len(rewards_avg))], [rewards_avg[i] + rewards_var[i] for i in range(len(rewards_avg))], label='avg + var', color='C1')

    plt.savefig("./images/episode_lengths_{}.png".format(file_suffix))


def plot_state_scalars(state_rewards_D, axl, file_suffix="NashLearner"):
    state_rewards_D = state_rewards_D[0]
    states = state_rewards_D.keys()
    for s in states:
        axl[s].set_title("State {}".format(s))
        axl[s].plot(
            [i for i in range(len(state_rewards_D[s][:-300]))],
            state_rewards_D[s][:-300],
            label=file_suffix,
        )
        axl[s].legend(loc="upper right")

    # axl[max(list(states))].set_xlabel('Actions per episode * episodes -->')
    # fig.set_ylabel('Rewards for defender -->')


def save_data(data, file_name="tmp.data"):
    with open("./outputs/{}.pickle".format(file_name), "wb") as f:
        pickle.dump(data, f)


def learn(env, learner=NashLearner, num_try=1):
    episode_lengths = []
    state_rewards_for_D = []
    distance_to_optimal_policy = []

    opt_pi = None
    # Optimal policy in the case of IDS
    # TODO: Make this a sub-routine in the actual game
    # opt_pi = {
    #     0: {"pi_lost": 1.0},
    #     1: {"pi_mon-LDAP": 0.6, "pi_no-mon": 0.4},
    #     2: {"pi_mon-FTP": 0.461, "pi_mon-Web": 0.539, "pi_no-mon": 0.0,},
    #     3: {"pi_mon-FTP": 1.0, "pi_no-mon": 0.0},
    # }
    opt_pi = {
        0: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0},
        1: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0},
        2: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0},
        3: {'pi_0': 0.5, 'pi_1': 0.5, 'pi_2': 0.0, 'pi_3': 0.0}
    }

    for t in range(num_try):
        rl_agent = learner(env, discount_factor=0.6, alpha=0.05)
        el, srd, dto = run(env, rl_agent, episodes=140, optimal_policy=opt_pi)
        episode_lengths.append(group_episode_rewards(el))
        state_rewards_for_D.append(srd)
        distance_to_optimal_policy.append(dto)

    save_data(
        (episode_lengths, state_rewards_for_D, distance_to_optimal_policy),
        file_name="exp_data_{}".format(rl_agent.get_name()),
    )

    # plot_state_scalars(state_rewards_for_D, scalar='rewards', file_suffix=rl_agent.get_name())
    # print(distance_to_optimal_policy)

    return episode_lengths, state_rewards_for_D, distance_to_optimal_policy


if __name__ == "__main__":
    env = Game()
    fig, axl = plt.subplots(
        len(env.S), sharex=True, figsize=(8, 7.5), gridspec_kw={"hspace": 0.4}
    )

    try:
        episode_lengths, state_rewards_for_D, distance_to_optimal_policy = pickle.load(
            open("outputs/exp_data_NashLearner.pickle", "rb")
        )
    except:
        episode_lengths, state_rewards_for_D, distance_to_optimal_policy = learn(
            env, NashLearner, num_try=1
        )
    plot_state_scalars(state_rewards_for_D, axl, file_suffix="Nash")

    try:
        episode_lengths, state_rewards_for_D, distance_to_optimal_policy = pickle.load(
            open("outputs/exp_data_SSELearner.pickle", "rb")
        )
    except:
        episode_lengths, state_rewards_for_D, distance_to_optimal_policy = learn(
            env, StackelbergLearner, num_try=1
        )
    plot_state_scalars(state_rewards_for_D, axl, file_suffix="SSE")

    fig.text(0.5, 0.04, 'episodes * steps -->', ha='center')
    fig.text(0.04, 0.5, 'Defender\'s Reward in state s --> ', va='center', rotation='vertical')

    plt.savefig("./images/state_rewards.png")
    # plot_state_rewards(state_rewards_for_D, file_suffix=rl_agent.get_name())
    # plot_episode_lengths(episode_lengths, file_suffix=rl_agent.get_name())

    # rl_agent = StackelbergLearner(env, discount_factor=0.5, alpha=0.1)
    # run(env, rl_agent, episodes=50)

    # plt.savefig("./defender_rewards_mtd.png")
