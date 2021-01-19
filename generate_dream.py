import MDRNN
import os
import numpy as np
import argparse
import tensorflow as tf

Z_DIM = 32
ACTION_DIM = 3
IN_DIM = Z_DIM + ACTION_DIM
OUT_DIM = Z_DIM

LSTM_UNITS = 256
N_MIXES = 5

DIR_NAME = './data/rollout/'

def main(args):

    frames = int(args.frames)

    mdrnn = MDRNN.MDRNN(IN_DIM, OUT_DIM, LSTM_UNITS, N_MIXES)
    try:
        mdrnn.set_weights('./MDRNN/weights.h5')
    except:
        print("Ensure ./MDRNN/weights.h5 exists")
        raise


    z = np.load('./data/initial_z/0.npz')['starting_z']
    dream = [z]

    h = tf.zeros([1, LSTM_UNITS])
    c = tf.zeros([1, LSTM_UNITS])

    a = tf.reshape([-0.1,1,0], [1,ACTION_DIM]) # ACCELERATE

    for i in range(frames):
        z = tf.concat([z,a], axis = 1)
        z = tf.reshape(z, [1,1, IN_DIM])
        z, h, c = mdrnn.forward_sample(z, h, c)
        dream.append(z)

    np.savez_compressed('./data/dreams/0', dream=dream)
    return mdrnn



if __name__ == "__main__":
        parser = argparse.ArgumentParser(description=('Test RNN'))
        parser.add_argument('--frames',default = 120, help='number of frames to dream')

        args = parser.parse_args()

        mdrnn = main(args)


