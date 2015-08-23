#%%
import time
import math
import sys
import argparse
import cPickle as pickle
import copy
import os
import six

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from VRAE import VRAE, make_initial_state

import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str,   default="dataset")
parser.add_argument('--output_dir',     type=str,   default="model")
parser.add_argument('--dataset',        type=str,   default="midi")
parser.add_argument('--init_from',      type=str,   default="")
parser.add_argument('--clip_grads',     type=int,   default=5)
parser.add_argument('--gpu',            type=int,   default=-1)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


if args.dataset == 'midi':
    midi = dataset.load_midi_data('%s/midi/sample.mid' % args.data_path)
    train_x = midi[:120].astype(np.float32)

    n_x = train_x.shape[1]
    n_hidden = [500]
    n_z = 2
    n_y = n_x

    frames  = train_x.shape[0]
    n_batch = 6
    seq_length = frames / n_batch

    split_x = np.vsplit(train_x, n_batch)

    n_epochs = 500
    continuous = False


if args.dataset == 'bvh':
    frames, frame_time, motion_data = dataset.load_bvh_data("%s/bvh/sample.bvh")
    max_motion = np.max(motion_data, axis=0)
    min_motion = np.min(motion_data, axis=0)

    norm_motion_data = (motion_data - min_motion) / (max_motion - min_motion)
    train_x = norm_motion_data
    train_y = norm_motion_data

    n_x = train_x.shape[1]
    n_hidden = [250]
    n_z = 10
    n_y = n_x

    n_online= 10
    n_batch = train_x.shape[0] / n_online

    if train_x.shape[0] % n_online != 0:
        reduced_sample = train_x.shape[0] % n_online
        train_x = train_x[:train_x.shape[0] - reduced_sample]

    n_epochs = 500
    continuous = True



n_hidden_recog = n_hidden
n_hidden_gen   = n_hidden
n_layers_recog = len(n_hidden_recog)
n_layers_gen   = len(n_hidden_gen)

layers = {}

# Recognition model.
rec_layer_sizes = [(train_x.shape[1], n_hidden_recog[0])]
rec_layer_sizes += zip(n_hidden_recog[:-1], n_hidden_recog[1:])
rec_layer_sizes += [(n_hidden_recog[-1], n_z)]

layers['recog_in_h'] = F.Linear(train_x.shape[1], n_hidden_recog[0], nobias=True)
layers['recog_h_h']  = F.Linear(n_hidden_recog[0], n_hidden_recog[0])

layers['recog_mean'] = F.Linear(n_hidden_recog[-1], n_z)
layers['recog_log_sigma'] = F.Linear(n_hidden_recog[-1], n_z)

# Generating model.
gen_layer_sizes = [(n_z, n_hidden_gen[0])]
gen_layer_sizes += zip(n_hidden_gen[:-1], n_hidden_gen[1:])
gen_layer_sizes += [(n_hidden_gen[-1], train_x.shape[1])]

layers['z'] = F.Linear(n_z, n_hidden_gen[0])
layers['gen_in_h'] = F.Linear(train_x.shape[1], n_hidden_gen[0], nobias=True)
layers['gen_h_h']  = F.Linear(n_hidden_gen[0], n_hidden_gen[0])

layers['output']   = F.Linear(n_hidden_gen[-1], train_x.shape[1])

if args.init_from == "":
    model = VRAE(**layers)
else:
    model = pickle.load(open(args.init_from))

# state pattern
state_pattern = ['recog_h', 'gen_h']

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()


# use Adam
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

total_losses = np.zeros(n_epochs, dtype=np.float32)

for epoch in xrange(1, n_epochs + 1):
    print('epoch', epoch)

    t1 = time.time()
    total_rec_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0
    outputs = np.zeros(train_x.shape, dtype=np.float32)
    # state = make_initial_state(n_hidden_recog[0], state_pattern)
    for i in xrange(n_batch):
        state = make_initial_state(n_hidden_recog[0], state_pattern)
        x_batch = split_x[i]

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)

        output, rec_loss, kl_loss, state = model.forward_one_step(x_batch, state, continuous, nonlinear_q='tanh', nonlinear_p='tanh', output_f = 'sigmoid', gpu=-1)

        outputs[i*seq_length:(i+1)*seq_length, :] = output

        loss = rec_loss + kl_loss
        total_loss += loss
        total_rec_loss += rec_loss
        total_losses[epoch-1] = total_loss.data

        optimizer.zero_grads()
        loss.backward()
        loss.unchain_backward()
        optimizer.clip_grads(args.clip_grads)
        optimizer.update()

    saved_output = outputs

    print "{}/{}, train_loss = {}, total_rec_loss = {}, time = {}".format(epoch, n_epochs, total_loss.data, total_rec_loss.data, time.time()-t1)

    if epoch % 100 == 0:
        model_path = "%s/VRAE_%s_%d.pkl" % (args.output_dir, args.dataset, epoch)
        with open(model_path, "w") as f:
            pickle.dump(copy.deepcopy(model).to_cpu(), f)
