#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Desc: Bidirectional (Symmetrical) Deep Neural Networks
#
#       Crossmodal translation and multimodal embedding with deep neural networks
#
#       As described in: V. Vukotić, C. Raymond, and G. Gravier. Bidirectional Joint Representation Learning with Symmetrical Deep Neural Networks
#       for Multimodal and Crossmodal Applications. In Proceedings of the 2016 ACM International Conference on Multimedia Retrieval (ICMR), pages 
#       343–346. ACM, 2016. (Available here: https://hal.inria.fr/hal-01314302/document)
#
# Usage: ./bidnn.py --help
#

import time
import theano
import argparse
import numpy as np
import scipy.linalg
from os import system
import theano.tensor as T
from lasagne.init import GlorotUniform
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import sigmoid, tanh, rectify, linear
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, concat, get_all_params, get_output, get_all_param_values, set_all_param_values



class BiDNN:
    def __init__(self, conf):
        self.conf = conf

        if self.conf.act == "linear":
            self.conf.act = linear
        elif self.conf.act == "sigmoid":
            self.conf.act = sigmoid
        elif self.conf.act == "relu":
            self.conf.act = rectify
        elif self.conf.act == "tanh":
            self.conf.act = tanh
        else:
            raise ValueError("Unknown activation function", self.conf.act)

        input_var_first   = T.matrix('inputs1')
        input_var_second  = T.matrix('inputs2')
        target_var        = T.matrix('targets')

        # create network        
        self.autoencoder, encoder_first, encoder_second = self.__create_toplogy__(input_var_first, input_var_second)
        
        self.out = get_output(self.autoencoder)
        
        loss = squared_error(self.out, target_var)
        loss = loss.mean()
        
        params = get_all_params(self.autoencoder, trainable=True)
        updates = nesterov_momentum(loss, params, learning_rate=self.conf.lr, momentum=self.conf.momentum)
        
        # training function
        self.train_fn = theano.function([input_var_first, input_var_second, target_var], loss, updates=updates)
        
        # fuction to reconstruct
        test_reconstruction = get_output(self.autoencoder, deterministic=True)
        self.reconstruction_fn = theano.function([input_var_first, input_var_second], test_reconstruction)
        
        # encoding function
        test_encode = get_output([encoder_first, encoder_second], deterministic=True)
        self.encoding_fn = theano.function([input_var_first, input_var_second], test_encode)

        # utils
        blas = lambda name, ndarray: scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]
        self.blas_nrm2 = blas('nrm2', np.array([], dtype=float))
        self.blas_scal = blas('scal', np.array([], dtype=float))

        # load weights if necessary
        if self.conf.load_model is not None:
            self.load_model()

    def __create_toplogy__(self, input_var_first=None, input_var_second=None):
        # define network topology
        if (self.conf.rep % 2 != 0):
            raise ValueError("Representation size should be divisible by two as it's formed by combining two crossmodal translations", self.conf.rep)

        # input layers
        l_in_first  = InputLayer(shape=(self.conf.batch_size, self.conf.mod1size), input_var=input_var_first)
        l_in_second = InputLayer(shape=(self.conf.batch_size, self.conf.mod2size), input_var=input_var_second)

        # first -> second
        l_hidden1_first   = DenseLayer(l_in_first, num_units=self.conf.hdn, nonlinearity=self.conf.act, W=GlorotUniform())         # enc1
        l_hidden2_first   = DenseLayer(l_hidden1_first, num_units=self.conf.rep//2, nonlinearity=self.conf.act, W=GlorotUniform()) # enc2
        l_hidden2_first_d = DropoutLayer(l_hidden2_first, p=self.conf.dropout)
        l_hidden3_first   = DenseLayer(l_hidden2_first_d, num_units=self.conf.hdn, nonlinearity=self.conf.act, W=GlorotUniform())    # dec1
        l_out_first       = DenseLayer(l_hidden3_first, num_units=self.conf.mod2size, nonlinearity=self.conf.act, W=GlorotUniform()) # dec2

        if self.conf.untied:
            # FREE
            l_hidden1_second   = DenseLayer(l_in_second, num_units=self.conf.hdn, nonlinearity=self.conf.act, W=GlorotUniform())         # enc1
            l_hidden2_second   = DenseLayer(l_hidden1_second, num_units=self.conf.rep//2, nonlinearity=self.conf.act, W=GlorotUniform()) # enc2
            l_hidden2_second_d = DropoutLayer(l_hidden2_second, p=self.conf.dropout)
            l_hidden3_second   = DenseLayer(l_hidden2_second_d, num_units=self.conf.hdn, nonlinearity=self.conf.act, W=GlorotUniform())    # dec1
            l_out_second       = DenseLayer(l_hidden3_second, num_units=self.conf.mod1size, nonlinearity=self.conf.act, W=GlorotUniform()) # dec2
        else:
            # TIED middle
            l_hidden1_second   = DenseLayer(l_in_second, num_units=self.conf.hdn, nonlinearity=self.conf.act, W=GlorotUniform())             # enc1
            l_hidden2_second   = DenseLayer(l_hidden1_second, num_units=self.conf.rep//2, nonlinearity=self.conf.act, W=l_hidden3_first.W.T) # enc2
            l_hidden2_second_d = DropoutLayer(l_hidden2_second, p=self.conf.dropout)
            l_hidden3_second   = DenseLayer(l_hidden2_second_d, num_units=self.conf.hdn, nonlinearity=self.conf.act, W=l_hidden2_first.W.T) # dec1
            l_out_second       = DenseLayer(l_hidden3_second, num_units=self.conf.mod1size, nonlinearity=self.conf.act, W=GlorotUniform())  # dec2

        l_out = concat([l_out_first, l_out_second])

        return l_out, l_hidden2_first, l_hidden2_second

    def load_dataset(self, X=None):
        if self.conf.verbosity > 1:
            print "Loading dataset..."
        if X is None:
            self.X_train, self.tl = load_svmlight_file(self.conf.fname_in, dtype=np.float32, multilabel=False)
            # we're saving tl (target labels) just in case they exist and the user needs them - since
            # this is unsupervised learning, we completely ignore the labels and don't expect them to exist
        else:
            self.X_train = X
        
        self.X_train = self.X_train.todense()

        if (self.conf.mod1size + self.conf.mod2size) != self.X_train.shape[1]:
            raise ValueError("Provided dimensionality of 1st modality ("+str(self.conf.mod1size)+") and 2nd modality ("+str(self.conf.mod2size)+") " \
                             "does not sum to the dimensionality provided in the input file ("+str(self.X_train.shape[1])+")")

        # indices of missing modalities (stored for later)
        self.idxMissingFirst = []
        self.idxMissingSecond = []
        
        # generate training data for modality translation
        self.X_first = [] 
        self.X_second = []
        
        bothMissing = both = 0
        if self.conf.ignore_zeroes:
            # zeroes are not treated as missing modalities
            # I have no idea why this might be useful, but ok :D
            # since idxMissing* are left empty, this is the only
            # place where we should take care of this
            for i in range(self.X_train.shape[0]):
                both += 1
                self.X_first.append(np.ravel(self.X_train[i, :self.conf.mod1size]))
                self.X_second.append(np.ravel(self.X_train[i, self.conf.mod1size:]))
        else:
            # zero vectors are treated as missing modalities (default)
            for i in range(self.X_train.shape[0]):
                if not np.any(self.X_train[i, :self.conf.mod1size]): # first missing
                    if np.any(self.X_train[i, self.conf.mod1size:]): # second not missing
                        # second ok, need to reconstruct first
                        self.idxMissingFirst.append(i)
                    else:
                        bothMissing +=  1 # missing both
                else: # first ok
                    if not np.any(self.X_train[i, self.conf.mod1size:]): # second missing
                        self.idxMissingSecond.append(i)
                    else: #both ok -> use them to train translator
                        both += 1
                        self.X_first.append(np.ravel(self.X_train[i, :self.conf.mod1size]))
                        self.X_second.append(np.ravel(self.X_train[i, self.conf.mod1size:]))
            
        if self.conf.verbosity > 1:
            print "Both modalities present:",both, "\nMissing 1st:", len(self.idxMissingFirst), "\nMissing 2nd:",len(self.idxMissingSecond)
            print "Missing both modalities:", bothMissing, "\n"

        self.X_first = np.array(self.X_first)
        self.X_second = np.array(self.X_second)


    def save_output(self, X, epoch=None):
        # write output to file
        if epoch is not None:
            fname_out = self.conf.fname_out.replace('%e', str(epoch).zfill(5))
        else:
            fname_out = self.conf.fname_out.replace('%e', 'final')

        if self.conf.verbosity > 1:
            print "Saving output to", fname_out, "..."
        dump_svmlight_file(X, self.tl, fname_out)

    def save_model(self, epoch=None):
        if epoch is not None:
            fname = self.conf.save_model.replace('%e', str(epoch).zfill(5))
        else:
            fname = self.conf.save_model.replace('%e', 'final')

        if self.conf.verbosity > 1:
            print "Saving model to", fname
        np.savez(fname, *get_all_param_values(self.autoencoder))

    def load_model(self):
        if self.conf.verbosity > 1:
            print "Loading pretrained model from", self.conf.load_model
        with np.load(self.conf.load_model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        set_all_param_values(self.autoencoder, param_values)

    def exec_command(self, epoch=None):
        if epoch is not None:
            cmd = self.conf.exec_command.replace('%e', str(epoch).zfill(5))
        else:
            cmd = self.conf.exec_command.replace('%e', 'final')
        system(cmd)

    def __iterate_minibatches__(self, mod1, mod2, batchsize, shuffle=False):
        assert len(mod1) == len(mod2)
        if shuffle:
            indices = np.arange(len(mod1))
            np.random.shuffle(indices)
        for start_idx in range(0, len(mod1) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)

            yield mod1[excerpt], mod2[excerpt], np.hstack([mod2[excerpt], mod1[excerpt]])

    def __norm__(self, vec):
        # L2 norm
        vec = np.asarray(vec, dtype=float)
        vec_len = np.sum(np.abs(vec))
        vec_len = self.blas_nrm2(vec)
        if vec_len > 0.0:
            return self.blas_scal(1.0 / vec_len, vec)
        return vec

    def train(self):
        # Train 
        for epoch in range(self.conf.epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.__iterate_minibatches__(self.X_first, self.X_second, self.conf.batch_size, shuffle=True):
                inputs_first, inputs_second, targets = batch
        
                train_err += self.train_fn(inputs_first, inputs_second, targets)
                train_batches += 1

            if self.conf.verbosity > 2:
                print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.conf.epochs, time.time() - start_time)),
                print("training loss:\t{:.6f}".format(train_err / train_batches))

            if self.conf.write_after is not None and (epoch+1) % self.conf.write_after == 0:
                out = self.predict()
                self.save_output(out, epoch+1)

                if self.conf.save_model is not None:
                    self.save_model(epoch = epoch+1)

                if self.conf.exec_command is not None:
                    self.exec_command(epoch = epoch+1)

    def predict(self):
        # generate either multimodal embedding or filled in missing modalities

        if not self.conf.crossmodal:
            # multimodal embedding

            # encode
            X_first  = self.X_train[:, self.conf.mod1size:]
            X_second = self.X_train[:, :self.conf.mod1size]
            enc_first, enc_second = self.encoding_fn(X_second, X_first)
            if self.conf.l2_norm:
                enc_first = self.__norm__(enc_first)
                enc_second = self.__norm__(enc_second)
            
            X_new = np.zeros([self.X_train.shape[0], self.conf.rep])
            
            for i in range(self.X_train.shape[0]):
                if i in self.idxMissingFirst: # first is missing, take from second
            
                    if i in self.idxMissingSecond: # second is ALSO missing
                        pass # heh ?
                    else: # SECOND OK, first missing
                        X_new[i, :] = np.hstack([enc_second[i,:], enc_second[i,:] ])
                else:
                    if i in self.idxMissingSecond: # FIRST OK, 2nd missing
                        X_new[i, :] = np.hstack([enc_first[i,:], enc_first[i,:] ])
                    else: # both OK
                        X_new[i, :] = np.hstack([enc_first[i,:], enc_second[i] ])


        else:
            # crossmodal expansion
            X_first_r  = self.X_train[:, self.conf.mod1size:]
            X_second_r = self.X_train[:, :self.conf.mod1size]
        
            reconstructed_out = self.reconstruction_fn(X_second_r, X_first_r)

            X_new = self.X_train # just reconstructing the missing modalities, others are left untouched
                    
            for i in range(X_new.shape[0]):
                if i in self.idxMissingFirst: # first is missing, take from second
            
                    if i in self.idxMissingSecond: # second is ALSO missing
                        pass # heh ?
                    else: # SECOND OK, first missing
                        if self.conf.l2_norm:
                            X_new[i, :self.conf.mod1size] = self.__norm__(reconstructed_out[i, self.conf.mod1size:])
                        else:
                            X_new[i, :self.conf.mod1size] = reconstructed_out[i, self.conf.mod1size:]
                else:
                    if i in self.idxMissingSecond: # FIRST OK, 2nd missing
                        if self.conf.l2_norm:
                            X_new[i, self.conf.mod1size:] = self.__norm__(reconstructed_out[i, :self.conf.mod1size])
                        else:
                            X_new[i, self.conf.mod1size:] = reconstructed_out[i, :self.conf.mod1size]
                   # else: # both OK

        return X_new

        
    


class Config:
    def __init__(self):
        self.hdn = None
        self.rep = None
        self.fname_in = None
        self.mod2size = None
        self.mod1size = None
        self.fname_out = None

        self.lr = 0.1
        self.act = "tanh"
        self.epochs = 1000
        self.verbosity = 3
        self.dropout = 0.2
        self.momentum = 0.9
        self.untied = False
        self.l2_norm = False
        self.batch_size = 128
        self.load_model = None
        self.save_model = None
        self.write_after = None
        self.crossmodal = False
        self.exec_command = None
        self.ignore_zeroes = False
        

if __name__ == "__main__":
    conf = Config()
    ap = argparse.ArgumentParser()

    # mandatory arguments
    ap.add_argument("infile", help="input file containing data in libsvm format")
    ap.add_argument("outfile", help="output file where the multimodal representation is saved in libsvm format")
    ap.add_argument("mod1size", type=int, help="dimensionality of 1st modality")
    ap.add_argument("mod2size", type=int, help="dimensionality of 2nd modality") 
    ap.add_argument("hdnsize", type=int, help="dimensionality of 1st hidden layer")
    ap.add_argument("repsize", type=int, help="output (multimodal) representation dimensionality (2 * 2nd hdn layer dim)")

    # additional arguments
    ap.add_argument("-a", "--activation", help="activation function (default: tanh)")
    ap.add_argument("-e", "--epochs", type=int, help="epochs to train (0 to skip training and solely predict)")
    ap.add_argument("-d", "--dropout", type=float, help="dropout value (default: 0.2; 0 -> no dropout")
    ap.add_argument("-b", "--batch-size", help="batch size (default: 128)")
    ap.add_argument("-r", "--learning-rate", type=float, help="learning rate (default: 0.1)")
    ap.add_argument("-m", "--momentum", type=float, help="momentum (default: 0.9)")
    ap.add_argument("-l", "--load-model", help="load pretrained model from file")
    ap.add_argument("-s", "--save-model", help="save trained model to file (at the end or after WRITE_AFTER epochs); any %%e " \
                                               "will be replaced with the current epoch")
    ap.add_argument("-c", "--crossmodal", action="store_true", help="perform crossmodal expansion (fill in missing modalities) " \
                                                                    "instead of multimodal embedding")
    ap.add_argument("-n", "--l2-norm", action="store_true", help="L2 normalize output (representation or reconstruction)")
    ap.add_argument("-z", "--ignore-zeroes", action="store_true", help="do not treat zero vectors as missing modalities")
    ap.add_argument("-u", "--untied", action="store_true", help="do not tie the variables in the central part")
    ap.add_argument("-w", "--write-after", type=int, help="write prediction file after x epochs (not only in the end); use %%e " \
                                                          "in [outfile] to indicate epoch number")
    ap.add_argument("-x", "--exec-command", help="execute command after WRITE_AFTER epochs (-w) or after training; any %%e in "  \
                                                 "the command will be replaced with the current epoch")
    ap.add_argument("-v", "--verbosity", type=int, help="sets verbosity (0-3, default: 3)")
    args = ap.parse_args()


    conf.fname_in = args.infile
    conf.fname_out = args.outfile
    conf.mod1size = args.mod1size
    conf.mod2size = args.mod2size
    conf.hdn = args.hdnsize
    conf.rep = args.repsize

    if args.epochs is not None: # 0 epochs is also possible
        conf.epochs = args.epochs

    if args.activation:
        conf.act = args.activation

    if args.dropout:
        conf.dropout = args.dropout

    if args.batch_size:
        conf.batch_size = args.batch_size

    if args.learning_rate:
        conf.lr = args.learning_rate

    if args.momentum:
        conf.momentum = args.momentum

    if args.load_model:
        conf.load_model = args.load_model

    if args.save_model:
        conf.save_model = args.save_model

    if args.ignore_zeroes:
        conf.ignore_zeroes = True

    if args.untied:
        conf.untied = True

    if args.crossmodal:
        conf.crossmodal = True

    if args.l2_norm:
        conf.l2_norm = True

    if args.write_after:
        conf.write_after = args.write_after

    if args.exec_command:
        conf.exec_command = args.exec_command

    if args.verbosity is not None:
        conf.verbosity = args.verbosity

    if conf.verbosity > 0:
        print ""
        print "Input file :", conf.fname_in
        print "Output file:", conf.fname_out
        print "Size of 1st modality:", conf.mod1size
        print "Size of 2nd modality:", conf.mod2size
        print "1st hidden layer size:", conf.hdn
        print "Representation size:", conf.rep
        print ""

        print "Epochs:", conf.epochs
        print "Activation:", conf.act
        print "Dropout:", conf.dropout
        print "Learning rate:", conf.lr
        print "Momentum", conf.momentum
        print "Load:", conf.load_model
        print "Save:", conf.save_model
        print "Ignore zeores:", conf.ignore_zeroes
        print "Do not tie:", conf.untied
        print "L2 normalizing outputs:", conf.l2_norm
        print "Saving each x epochs:", conf.write_after
        print "Executing after x epochs:", conf.exec_command
        if conf.crossmodal:
            print "Performing crossmodal expansion"
        else:
            print "Performing multimodal embedding"
        print ""


    bidnn = BiDNN(conf)
    bidnn.load_dataset()
    bidnn.train()

    out = bidnn.predict()
    bidnn.save_output(out)

    if conf.save_model is not None:
        bidnn.save_model()

    if conf.exec_command is not None:
        bidnn.exec_command()
