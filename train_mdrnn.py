import MDRNN
import os
import numpy as np
import argparse

ROOT_DIR_NAME = './data/'
SERIES_DIR_NAME = './data/series/'

Z_DIM = 32
ACTION_DIM = 3

IN_DIM = Z_DIM + ACTION_DIM
OUT_DIM = Z_DIM
LSTM_UNITS = 256
N_MIXES = 5

def get_filelist(N):
    filelist = os.listdir(SERIES_DIR_NAME)
    filelist = [x for x in filelist if (x!= '.DS_STORE' and x!='.gitignore')]
    filelist.sort()
    length_filelist = len(filelist)

    if length_filelist > N:
        filelist = filelist[:N]

    if length_filelist < N:
        N = length_filelist

    return filelist, N

def random_batch(filelist, batch_size):
    """open batch_size files from filelist"""
    N_data = len(filelist)
    indices = np.random.permutation(N_data)[0:batch_size]

    z_list = []
    action_list = []
    rew_list = []
    done_list = []

    for i in indices:
        try:
            new_data = np.load(SERIES_DIR_NAME + filelist[i], allow_pickle=True)

            # this is the latent distribution
            mu = new_data['mu']
            log_var = new_data['log_var']
            action = new_data['action']
            reward = new_data['reward']
            done = new_data['done']

            reward = np.expand_dims(reward, axis=1)
            done = np.expand_dims(done, axis=1)

            s = log_var.shape

            z = mu + np.exp(log_var/2.0) * np.random.randn(*s)

            z_list.append(z)
            action_list.append(action)
            rew_list.append(reward)
            done_list.append(done)

        except Exception as e:
            print(e)


    z_list = np.array(z_list)
    action_list = np.array(action_list)
    rew_list = np.array(rew_list)
    done_list = np.array(done_list)

    return z_list, action_list, rew_list, done_list

def main(args):

    new_model = args.new_model
    N = int(args.N)
    steps = int(args.steps)
    batch_size = int(args.batch_size)

    mdrnn = MDRNN.MDRNN(IN_DIM, OUT_DIM, LSTM_UNITS, N_MIXES)

    if not new_model:
        try:
            mdrnn.set_weights('./MDRNN/weights.h5')
        except:
            print("Either set --new_model or ensure ./MDRNN/weights.h5 exists")
            raise

    filelist, N = get_filelist(N)

    for step in range(steps):
        print('STEP' + str(step))

        z, action, rew, done = random_batch(filelist, batch_size)
        rnn_in = np.concatenate([z[:, :-1, :], action[:, :-1, :]], axis = 2)
        rnn_out = z[:, 1:, :]

        mdrnn.train(rnn_in, rnn_out)

        if step % 10 == 0:
            mdrnn.model.save_weights('./MDRNN/weights.h5')
            print("Saved weights")

    mdrnn.model.save_weights('./MDRNN/weights.h5')
    print("Saved weights")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description=('Train RNN'))
        parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
        parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
        parser.add_argument('--steps', default = 4000, help='how many rnn batches to train over')
        parser.add_argument('--batch_size', default = 100, help='how many episodes in a batch?')

        args = parser.parse_args()

        main(args)


