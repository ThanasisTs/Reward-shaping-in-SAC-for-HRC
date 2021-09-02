import sys
import yaml

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
	game = Game()

	# create SAC agent
	sac = getSACAgent(config, game)

	# create experiment
	experiment = Experiment(game, sac, config=config)
	experiment.run()



if __name__ == "__main__":
	main()