from sac.sac_discrete.sac_discrete_agent import DiscreteSACAgent


def getSACAgent(config, env):
	buffer_max_size = config['Experiment']['buffer_max_size']
	update_interval = config['Experiment']['learn_every_n_episodes']
	scale = config['Experiment']['reward_scale']
	n_actions = env.action_space.n_actions
	# chkpt_dir = config['Experiment']['chkpt_dir']
	chkpt_dir = 'tmp/'
	scale = config['Experiment']['reward_scale']

	return DiscreteSACAgent(config=config, env=env, input_dims=env.observation_space,
		n_actions=n_actions, chkpt_dir=chkpt_dir, buffer_max_size=buffer_max_size, update_interval=update_interval,
		reward_scale=scale)


