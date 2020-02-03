from agents import *
from environments.ids_place_game import GeneralSum_Game
import numpy as np
import matplotlib.pyplot as plt


def sample_act_from_policy(pi, epsilon=0):
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


def run(episodes=500):
    env = GeneralSum_Game()

    rl_agent = NashLearner(env, discount_factor=0.5, alpha=0.1)
    rl_agent.initial_policy()
    s_t = env.get_start_state()

    episodes = [i for i in range(episodes)]
    rewards_D = []
    # rewards_A = []

    for i in episodes:
        # Sample a policy to execute in state s_t
        pi_D, pi_A = rl_agent.get_policy_in_state(s_t)
        a_D = sample_act_from_policy(pi_D)
        a_A = sample_act_from_policy(pi_A)

        # Act according to sampled policy
        r_D, r_A, s_next = env.act(s_t, a_D, a_A)
        rewards_D.append(r_D)

        rl_agent.update_Q(s_t, a_D, a_A, r_D, r_A, s_next)
        rl_agent.update_value_and_policy(s_t)

        s_t = s_next
        if env.is_end(s_t):
            break

    plt.plot(episodes[:i+1], rewards_D, marker='x')
    plt.savefig("./defender_rewards.png")


if __name__ == "__main__":
    run()
