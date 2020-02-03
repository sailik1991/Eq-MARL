from agents import *
from environments.ids_place_game import GeneralSum_Game
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
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    return np.random.choice(acts, p=probs)


def run(env, rl_agent, episodes=30):
    rl_agent.initial_policy()

    rewards_D = {}
    for i in range(episodes):
        j = 0
        s_t = env.get_start_state()
        while j < 100 or not env.is_end(s_t):
            j += 1

            # Sample a policy to execute in state s_t
            pi_D, pi_A = rl_agent.get_policy_in_state(s_t)
            a_D = sample_act_from_policy(pi_D)
            a_A = sample_act_from_policy(pi_A)

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
    rl_agent.print_Q()
    rl_agent.print_policy()

    states = rewards_D.keys()
    for s in states:
        if env.is_end(s):
            continue
        axl[s - 1].set_title("State {}".format(s))
        axl[s - 1].plot([i for i in range(len(rewards_D[s]))], rewards_D[s], label=rl_agent.get_name())
        axl[s - 1].legend(loc='upper right')


if __name__ == "__main__":
    env = GeneralSum_Game()
    fig, axl = plt.subplots(len(env.start_S), sharex=True, gridspec_kw={"hspace": 0.4})

    rl_agent = NashLearner(env, discount_factor=0.5, alpha=0.1)
    run(env, rl_agent)

    rl_agent = StackelbergLearner(env, discount_factor=0.5, alpha=0.1)
    run(env, rl_agent)

    plt.savefig("./defender_rewards.png")
