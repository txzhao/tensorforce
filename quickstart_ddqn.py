# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt

from tensorforce.agents import DDQNAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# solve GPU out-of-memory issue
os.environ['CUDA_VISIBLE_DEVICES'] = ""

max_episodes = 400
max_timesteps = 200
num_of_runs = 3
cur_rwd = []
aver_rwd = np.array((max_episodes, ))

# repetitive run
for run_no in range(1, num_of_runs+1):

	# Create an OpenAIgym environment
	env = OpenAIGym('CartPole-v0', visualize=False)		#TODO: cannot visualize through putty

	# Network as list of layers
	network_spec = [
		dict(type='dense', size=64, activation='relu'),
		dict(type='dense', size=32, activation='relu')
	]

	agent = DDQNAgent(
		states_spec=env.states,
		actions_spec=env.actions,
		network_spec=network_spec,
		# Agent
		explorations_spec= dict(
			type='epsilon_anneal',
			initial_epsilon=0.8,
			final_epsilon=0.01,
			timesteps=1e4
			),
		batched_observe=1,
		# MemoryAgent
		batch_size=64,
		memory = dict(
			type='prioritized_replay',
			capacity=50000,
			prioritization_weight=0.2,
			prioritization_constant=1e-6
		),
		first_update=1000,
		update_frequency=1,
		repeat_update=1,
		# DQNAgent
		optimizer=dict(
			type='clipped_step',
			clipping_value=0.005,
			optimizer=dict(
				type='adam',
				learning_rate=5e-4
			)
		),
		#optimization_steps=10,
		# Model
		variable_noise=None,
		discount=0.99,
		#double_q_model=True,
		#huber_loss=1.0,
		# DistributionModel
		target_sync_frequency=500,
		target_update_weight=1.0
	)

	# Create the runner
	runner = Runner(agent=agent, environment=env)

	# Callback function printing episode statistics
	def episode_finished(r):
		print("Run {run} - Finished episode {ep} after {ts} timesteps (reward: {reward})"
			.format(ep=r.episode, ts=r.episode_timestep, reward=r.episode_rewards[-1], run=run_no))
		return True

	# Start learning
	runner.run(episodes=max_episodes, max_episode_timesteps=max_timesteps, episode_finished=episode_finished)

	# Print statistics
	print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
		ep=runner.episode,
		ar=np.mean(runner.episode_rewards[-100:]))
	)

	cur_rwd = runner.episode_rewards
	tmp_cur_rwd = np.array(cur_rwd)
	aver_rwd = aver_rwd + (tmp_cur_rwd - aver_rwd)/float(run_no)
	cur_rwd = list(tmp_cur_rwd)


# Save reward plot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(range(len(aver_rwd)), list(aver_rwd))
ax.set_title('Number of Run: ' + str(num_of_runs))
ax.set_xlabel('Episodes')
ax.set_ylabel('Average Rewards')
fig.savefig('/Users/txzhao/Desktop/thesis/tensorforce/figs/result.png')
plt.close(fig)

