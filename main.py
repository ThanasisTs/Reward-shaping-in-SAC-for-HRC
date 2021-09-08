import sys
import yaml
import time

from sac_env import *
from utils import *
from experiment import *


def get_config():
	try:
		file = open(sys.argv[1])
		config = yaml.safe_load(file)
		return config
	except Exception as e:
		print(e)
		sys.exit()


def main():
	# YAML parse
	config = get_config()

	# create game environment
	game = Game(controlled_variable=config['Game']['controlled_variable'])

	if config['Game']['save']:
		rl_model_dir, chkpt_dir, plot_dir, load_checkpoint_name = get_plot_and_chkpt_dir(config)

	# create SAC agent
	sac = getSACAgent(config, game, rl_model_dir)

	start_experiment_time = time.time()

	# create experiment
	experiment = Experiment(game, sac, config=config)
	experiment.run()

	end_experiment_time = time.time()

	experiment_duration = start_experiment_time - end_experiment_time
	print(f'Total Experiment time: {experiment_duration:.4f}')



if __name__ == "__main__":
	main()