from agents import StackelbergLearner, URSLearner
from environments.mtd_web_apps.env import Game

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import math
from pandas import DataFrame

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(state_rewards):
    df = DataFrame(state_rewards,
                   columns=['agent', 'state', 'try', 'episodes ->', 'Defender\'s reward ->'])

    plt.figure(figsize=(16, 7.5))
    g = sns.relplot(x="episodes ->", y="Defender\'s reward ->", col="state", hue="agent", style="agent", kind="line", data=df)
    plt.savefig("./images/state_rewards.png")


def get_data(agent='SSE', data_len=100):
    episode_lengths, state_rewards_for_D, distance_to_optimal_policy = pickle.load(
        open("outputs/exp_data_{}Learner.pickle".format(agent), "rb")
    )

    state_rewards = []
    for trial in range(len(state_rewards_for_D)):
        for state, reward_list in state_rewards_for_D[trial].items():
            for eps in range(min(data_len, len(reward_list))):
                try:
                    state_rewards.append([agent, state, trial, eps, reward_list[eps]])
                except KeyError:
                    state_rewards[state] = [[agent, state, trial, eps, reward_list[eps]]]

    return state_rewards

if __name__ == "__main__":
    env = Game()

    ''' Set graph variables '''
    sns.set()
    sns.set_context("paper")  # options: paper, talk, posters
    # sns.set_palette("deep")
    flatui = ["#47697E", "#5B7444", "#FFCC33"]
    sns.set_palette(flatui)

    rewards = get_data('SSE')
    rewards += get_data('URS')

    plot_rewards(rewards)

    # # {k: v for d in state_rewards_for_D for k, v in d.items()}
    #
    # df = DataFrame(np.array([[1, 1, 10], [1, 2, 11], [2, 1, 12], [2, 2, 6]]),
    #                columns=['try', 'eps', 'reward'])
    # print(df)
    #
    # plt.figure(figsize=(20, 9))
    # sns.lineplot(data=df, x='eps', y='reward')
    # plt.savefig("./demo.png")
    #
    # exit(1)
    #
    # states_1 = []
    # # for d in state_rewards_for_D:
    #
    #
    # # print(state_rewards_for_D)
    # # reward_dict = {k: v for d in state_rewards_for_D for k, v in d.items()}
    # reward_dict = {1:10, 1:11, 2:3, 2:4}
    # df = DataFrame(list(reward_dict.items()), columns=['States', 'Rewards'])
    # print(df)
    #
    # exit(1)
    #
    # plot_state_scalars(state_rewards_for_D, axl, file_suffix=eq)
    # fig.text(0.015, 0.5, "R (defender) -->", va="center", rotation="vertical")
    # fig.text(0.5, 0.04, "episodes -->", ha="center")
    # plt.savefig("./images/state_rewards.png")
    #
    # fig, axl = None, None
    # if PLOT_OPT_DIST or PLOT_R:
    #     fig, axl = plt.subplots(
    #         len(env.S), sharex=True, figsize=(7, 8.5), gridspec_kw={"hspace": 0.4}
    #     )