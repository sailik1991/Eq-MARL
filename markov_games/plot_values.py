from agents import StackelbergLearner, URSLearner
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
import numpy as np
import pickle


def plot_rewards(state_rewards):
    df = DataFrame(
        state_rewards,
        columns=["agent", "state", "try", "episodes ->", "Defender's reward ->"],
    )

    plt.figure(figsize=(16, 7.5))
    g = sns.relplot(
        x="episodes ->",
        y="Defender's reward ->",
        col="state",
        hue="agent",
        style="agent",
        kind="line",
        data=df,
    )
    plt.savefig("./images/state_rewards.png")


def get_data(agent="SSE", data_len=100):
    episode_lengths, state_rewards_for_D, distance_to_optimal_policy = pickle.load(
        open("outputs/exp_data_{}Learner.pickle".format(agent), "rb")
    )

    legends = {
        "URS": "URS",
        "SSE": "BSS-Q"
    }

    state_rewards = []
    for trial in range(len(state_rewards_for_D)):
        for state, reward_list in state_rewards_for_D[trial].items():
            for eps in range(min(data_len, len(reward_list))):
                try:
                    agent_name = legends[agent]
                except KeyError:
                    agent_name = "{}-Q".format(agent)
                try:
                    state_rewards.append([agent_name, state, trial, eps, reward_list[eps]])
                except KeyError:
                    state_rewards[state] = [
                        [agent_name, state, trial, eps, reward_list[eps]]
                    ]

    return state_rewards


if __name__ == "__main__":
    """ Set graph variables """
    sns.set()
    sns.set_context("paper")  # options: paper, talk, posters
    sns.set_palette("deep")

    rewards = get_data("SSE")
    rewards += get_data("URS")
    rewards += get_data("EXP")
    rewards += get_data("Nash")

    plot_rewards(rewards)
