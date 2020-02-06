from agents import *
# from environments.ids_place_game import Game
from environments.mtd_switching_costs import Game
import numpy as np
import matplotlib.pyplot as plt


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


def run(env, rl_agent, episodes=25):

    max_steps_per_episode = 100
    exploration_rate_decay = 0.999

    # Initialize RL agent with Uniform Random Strategy
    rl_agent.initial_policy()
    rewards_D = {}

    for i in range(episodes):

        # Read discussion on decaying exploration rate.
        # https://ai.stackexchange.com/questions/7923/learning-rate-decay-and-exploration-rate-decay
        # As opposed to doing this after each action, we do this decay after every episode.
        # A lower bound ensure that if the agent isn't stuck with a back policy just because LR -> 0
        epsilon = max(0.1 * exploration_rate_decay, 0.03)
        exploration_rate_decay *= exploration_rate_decay

        j = 0
        s_t = env.get_start_state()
        while j < max_steps_per_episode and not env.is_end(s_t):
            j += 1

            # Sample a policy to execute in state s_t
            pi_D, pi_A = rl_agent.get_policy_in_state(s_t)
            a_D = sample_act_from_policy(pi_D, epsilon=epsilon)
            a_A = sample_act_from_policy(pi_A, epsilon=epsilon)

            # print('[DEBUG] Policy \nDef: {} \nAtt: {}'.format(pi_D, pi_A))
            # print('[DEBUG] State: {}, Def: {}, Att: {}'.format(s_t, a_D, a_A))

            # Act according to sampled policy
            r_D, r_A, s_next = env.act(s_t, a_D, a_A)

            # Save rewards obtained in a state for plotting
            try:
                rewards_D[s_t].append(r_D)
            except KeyError:
                rewards_D[s_t] = []

            rl_agent.update_Q(s_t, a_D, a_A, r_D, r_A, s_next)
            rl_agent.update_value_and_policy(s_t)

            s_t = s_next

    # [Debug] Ensure that the Q-values are approx. equal to true reward values.
    print('[DEBUG] {}\'s policy:'.format(rl_agent.get_name()))
    # rl_agent.print_Q()
    rl_agent.print_policy()

    states = rewards_D.keys()
    diff = len(states) - len(axl) + 1
    for s in states:
        if env.is_end(s):
            continue
        axl[s - diff].set_title("State {}".format(s))
        axl[s - diff].plot([i for i in range(len(rewards_D[s]))], rewards_D[s], label=rl_agent.get_name())
        axl[s - diff].legend(loc='upper right')


if __name__ == "__main__":
    env = Game()
    fig, axl = plt.subplots(len(env.start_S), sharex=True, gridspec_kw={"hspace": 0.4})

    rl_agent = NashLearner(env, discount_factor=0.5, alpha=0.1)
    run(env, rl_agent, episodes=50)

    rl_agent = StackelbergLearner(env, discount_factor=0.5, alpha=0.1)
    run(env, rl_agent, episodes=50)

    plt.savefig("./defender_rewards_mtd.png")
