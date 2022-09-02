##### Table of Contents  
- [Overview and Running the Code](#headers_1)

- [Citation](#headers_2)

If you use the code or derive code from it-- Cite the MIT License, Cite our work

<a name="headers_1"/>

## Overview and Running the Code

The code learns movement policies in three MTD domain. The conda environment with all the prerequisite dependencies can be found at:
```
./conda_env.yml
```
Running the BSS-Q (but not the baselines) requires one to have a Gurobi installed in their machine alongwith a Liscence Key (it is free for academic researchers). More information on this can be found [here](https://www.gurobi.com/documentation/9.0/quickstart_mac/retrieving_and_setting_up_.html). We will now briefly discuss how to use the existing data-files that are used in the paper and also how to generate all the data from scratch.

### Moving Target Defense for Placement of Intrusion Detection Systems

Note that this domain does not need the full generality offered by Bayesian Stackelberg Markov Games and can be handled via MARL approaches for Markov Games.

- Experimental data shown in the paper can be found at:

Hyperparameters -- `./markov_games/outputs/mtd-ids/hyperparameters.ini`

Saved Rewards -- `./markov_games/outputs/mtd-ids/exp_data_` `<learner name>` `Learner.pickle`

Saved Plot -- `./markov_games/images/mtd-ids/state_rewards.png`

- Reproducing plot with existing data.

```bash
cd ./markov_games

# Copy saved data files
cp ./outputs/mtd-ids/*.ini ./

# Run code to plot graph
python plot_values.py

# View output
eog ./images/mtd-ids/state_rewards.png
```

- Training RL agent and generating graph from scratch
```bash
cd ./markov_games

# Edit hyperparameters
vi hyperparameters.ini

# Run code (takes significant amount of time)
python run.py

# plot using generated data (stored in output folder)
python plot_values.py

# View output
eog ./images/mtd-ids/state_rewards.png
```

### Moving Target Defense for Web-Application Security

#### Switching costs factored into the rewards of the BSMG

- Experimental data shown in the paper can be found at:

Hyperparameters -- `./bayesian_stackelberg_markov_games/outputs/mtd_wa/paper/hyperparameters.ini`

Saved Rewards -- `./bayesian_stackelberg_markov_games/outputs/mtd_wa/paper/exp_data_` `<learner name>` `Learner.pickle`

Saved Plot -- `./bayesian_stackelberg_markov_games/images/mtd_wa/paper/state_rewards.png`

- Reproducing plot with existing data.

```bash
cd ./bayesian_stackelberg_markov_games

# Copy saved data files
cp ./outputs/mtd_wa/paper/*.ini ./

# Run code to plot graph
python plot_values.py

# View output
eog ./images/mtd_wa/paper/state_rewards.png
```

- Training RL agent and generating graph from scratch
```bash
cd ./bayesian_stackelberg_markov_games

# Edit hyperparameters
vi hyperparameters.ini

# Run code (takes significant amount of time)
python run.py

# plot using generated data (stored in output folder)
python plot_values.py

# View output
eog ./images/mtd_wa/paper/state_rewards.png
```

We have data from other runs that is available by simply replacing `paper` by `run_5_30`.

#### Switching factored into the transition dynamics of BSMG

- Experimental data shown in the paper can be found at:

Hyperparameters -- `./bayesian_stackelberg_markov_games/outputs/mtd_wa_stochastic/hyperparameters.ini`

Saved Rewards -- `./bayesian_stackelberg_markov_games/outputs/mtd_wa_stochastic/exp_data_` `<learner name>` `Learner.pickle`

Saved Plot -- `./bayesian_stackelberg_markov_games/images/mtd_wa_stochastic/state_rewards.png`

- Reproducing plot with existing data.

```bash
cd ./bayesian_stackelberg_markov_games

# Copy saved data files
cp ./outputs/mtd_wa_stochastic/*.ini ./

# Copy saved hyperparameters
cp ./outputs/mtd_wa_stochastic/*.pickle ./outputs/

# Run code to plot graph
python plot_values.py

# View output
eog ./images/mtd_wa_stochastic/state_rewards.png
```

- Training RL agent and generating graph from scratch
```bash
cd ./bayesian_stackelberg_markov_games

# Edit hyperparameters
vi hyperparameters.ini

# Run code (takes significant amount of time)
python run.py

# plot using generated data (stored in output folder)
python plot_values.py

# View output
eog ./images/mtd_wa_stochastic/state_rewards.png
```

<a name="headers_2"/>

## Citation
```
@article{sengupta2020multi,
  title={Multi-agent reinforcement learning in bayesian stackelberg markov games for adaptive moving target defense},
  author={Sengupta, Sailik and Kambhampati, Subbarao},
  journal={NeurIPS Workshop on Real-world Reinforcement Learning},
  year={2020}
}
```
