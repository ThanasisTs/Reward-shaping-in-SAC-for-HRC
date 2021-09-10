import os
import shutil
from sac.sac_discrete.sac_discrete_agent import DiscreteSACAgent

def getSACAgent(config, env, checkpoint_dir=None, save=False):
    buffer_max_size = config['Experiment']['buffer_max_size']
    update_interval = config['Experiment']['learn_every_n_episodes']
    scale = config['Experiment']['reward_scale']
    n_actions = env.action_space.n_actions
    chkpt_dir = config['Experiment']['chkpt_dir'] + '/' + checkpoint_dir
    scale = config['Experiment']['reward_scale']
    print(chkpt_dir)
    return DiscreteSACAgent(config=config, env=env, input_dims=env.observation_space,
        n_actions=n_actions, chkpt_dir=chkpt_dir, buffer_max_size=buffer_max_size, update_interval=update_interval,
        reward_scale=scale)


def get_save_dir(config):
    total_number_updates = config['Experiment']['total_update_cycles']
    participant = config['Game']['participant_name']
    learn_every = config['Experiment']['learn_every_n_episodes']
    save_chkpt_dir = str(int(total_number_updates / 1000)) + 'K_every' + str(learn_every) + '_' + '_' + participant
    
    i = 1
    while os.path.exists(save_chkpt_dir + '_' + str(i)):
        i += 1
    os.makedirs(save_chkpt_dir + '_' + str(i))
    save_chkpt_dir = save_chkpt_dir + '_' + str(i)

    return save_chkpt_dir


def get_plot_and_chkpt_dir(config):
    load_checkpoint, load_checkpoint_name = config['Game']['load_checkpoint'], config['Game']['checkpoint_name']

    total_number_updates = config['Experiment']['total_update_cycles']
    participant = config['Game']['participant_name']
    learn_every = config['Experiment']['learn_every_n_episodes']

    plot_dir = None
    if not load_checkpoint:
        rl_model_dir = str(int(total_number_updates / 1000)) + 'K_every' + str(learn_every) + '_' + '_' + participant
        chkpt_dir = 'data/'+ rl_model_dir
        plot_dir = 'plots/' + rl_model_dir
        i = 1
        while os.path.exists(chkpt_dir + '_' + str(i)):
            i += 1
        os.makedirs(chkpt_dir + '_' + str(i))
        chkpt_dir = chkpt_dir + '_' + str(i)
        j = 1
        while os.path.exists(plot_dir + '_' + str(j)):
            j += 1
        os.makedirs(plot_dir + '_' + str(j))
        plot_dir = plot_dir + '_' + str(j)
    else:
        print("Loading Model from checkpoint {}".format(load_checkpoint_name))
        chkpt_dir = load_checkpoint_name
    return rl_model_dir, chkpt_dir, plot_dir, load_checkpoint_name



