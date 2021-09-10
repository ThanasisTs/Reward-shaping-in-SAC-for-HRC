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


	# create SAC agent
	save_chkpt_dir = get_save_dir(config)
	sac = getSACAgent(config, game, save_chkpt_dir)

	# create trained SAC agent if asked
	if config['Game']['load_checkpoint']:
		trained_sac = getSACAgent(config, game, config['Game']['load_checkpoint_dir'])

	start_experiment_time = time.time()

	# create experiment
	experiment = Experiment(env=game, agent=sac, config=config, trained_agent=trained_sac)
	experiment.run()

	end_experiment_time = time.time()

	experiment_duration = start_experiment_time - end_experiment_time
	print(f'Total Experiment time: {experiment_duration:.4f}')



if __name__ == "__main__":
	main()