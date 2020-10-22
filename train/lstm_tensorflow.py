import argparse
import copy
import numpy as np
import os
import random
from sklearn.utils import shuffle
import tensorflow as tf
from time import time
try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops
from load import process_eod_data
from evaluator import evaluate

class LSTM:
    def __init__(self, data_path, market_name, parameters, model_path, model_save_path,
                 tra_end=5200, val_end=5900,
                 steps=1, epochs=50, batch_size=150, gpu=False, fix_init=0, reload=0):
        self.data_path = data_path
        self.market_name = market_name
        self.model_path = model_path
        self.model_save_path = model_save_path
        # model parameters
        self.paras = copy.copy(parameters)
        # training parameters
        self.steps = steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu
        if fix_init == 1:
            self.fix_init = True
        else:
            self.fix_init = False
        if reload == 1:
            self.reload = True
        else:
            self.reload = False

        self.tra_end = tra_end
        self.val_end = val_end

        #  load_data
        self.tra_pv, self.tra_gt,\
        self.val_pv, self.val_gt,\
        self.test_pv, self.test_gt = process_eod_data(data_path, parameters['seq'], self.tra_end, self.val_end)
        self.fea_dim = self.tra_pv.shape[2]
        print('train', self.tra_pv.shape)
        print('valid', self.val_pv.shape)

    def get_batch(self, sta_ind=None):  # 获得每批训练数据，大小batch_size
        if sta_ind is None:
            sta_ind = random.randrange(0, self.tra_pv.shape[0])
        if sta_ind + self.batch_size < self.tra_pv.shape[0]:
            end_ind = sta_ind + self.batch_size
        else:
            sta_ind = self.tra_pv.shape[0] - self.batch_size
            end_ind = self.tra_pv.shape[0]
        return self.tra_pv[sta_ind:end_ind, :, :], \
               self.tra_gt[sta_ind:end_ind, :]

    def construct_graph(self):
        print("constructing graph")
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('using device:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()
            if self.fix_init:
                tf.set_random_seed(123456)
            self.gt_var = tf.placeholder(tf.float32, [None, 1])
            self.pv_var = tf.placeholder(tf.float32, [None, self.paras['seq'], self.fea_dim])  # self.fea_dim load data补齐

            # set lstm unit
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.paras['unit'])

            # input FC layer
            # self.in_lat = tf.layers.dense(
            #     self.pv_var, units=self.fea_dim, activation=tf.nn.tanh, name='in_lat',
            #     kernel_initializer=tf.glorot_uniform_initializer()
            # )
            # lstm layer
            self.lstm_outputs, _ = tf.nn.dynamic_rnn(
                self.lstm_cell, self.pv_var, dtype=tf.float32
            )

            # FC predict
            self.pred = tf.layers.dense(
                self.lstm_outputs[:, -1, :], units=1, activation=tf.nn.sigmoid, name='pred',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            self.loss = tf.losses.log_loss(self.gt_var, self.pred)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.paras['lr']).minimize(self.loss)

    def train(self):
        self.construct_graph()
        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        bat_count = self.tra_pv.shape[0] // self.batch_size # 一个epoch内总批数
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1
        print(bat_count, self.batch_size)

        for i in range(self.epochs):
            t1 = time()
            tra_loss = 0.0
            acc = 0
            for j in range(bat_count):
                pv_b, gt_b = self.get_batch(j*self.batch_size)
                feed_dict = {
                    self.pv_var: pv_b,
                    self.gt_var: gt_b
                }

                cur_loss,cur_pred,_ = sess.run((self.loss,self.pred,self.optimizer), feed_dict)
                tra_loss += cur_loss
                cur_tra_perf = evaluate(cur_pred, gt_b)
                acc += cur_tra_perf['acc']
            print('>>>>> Training: train loss', tra_loss / bat_count,'train acc',acc /bat_count)


            feed_dict = {
                self.pv_var: self.val_pv,
                self.gt_var: self.val_gt
            }
            val_loss, val_pre= sess.run((self.loss, self.pred, ), feed_dict)
            cur_val_perf = evaluate(val_pre, self.val_gt)

            print('\tVal loss:', val_loss, "\tValid epoch", cur_val_perf)
            t2 = time()
            print('epoch:', i, ('time: %.4f ' % (t2 - t1)))
            self.tra_pv, self.tra_gt = shuffle(self.tra_pv, self.tra_gt, random_state=0)
        sess.close()
        tf.reset_default_graph()

    # def test(self):
    #     self.construct_graph()
    #     sess = tf.Session()
    #     saver = tf.train.Saver()
    #     if self.reload:
    #         saver.restore(sess, self.model_path)
    #         print('model restored')
    #     else:
    #         sess.run(tf.global_variables_initializer())
    #     feed_dict = {}


if __name__ == '__main__':
    desc = 'lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--path', help='path of data', type=str, default='../data/limit_daily/SSE')
    parser.add_argument('-m', '--market', help='market', type=str, default='SSE')

    parser.add_argument('-u', '--unit', help='number of hidden units in lstm', type=int, default=4)
    parser.add_argument('-r', '--learning_rate', help='learning rate', type=float, default=0.1)
    parser.add_argument('-l', '--seq', help='length of history', type=int, default=16)

    parser.add_argument('-q', '--model_path', help='path to load model', type=str, default='')
    parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model', default='')

    parser.add_argument('-s', '--step', help='steps to make prediction', type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=512)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=1000)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-o', '--action', type=str, default='train', help='train, test')
    parser.add_argument('-f', '--fix_init', type=int, default=0, help='use fixed initialization')
    parser.add_argument('-rl', '--reload', type=int, default=0, help='use pre-trained parameters')
    args = parser.parse_args()
    print(args)

    parameters = {
        'seq': int(args.seq),
        'unit': int(args.unit),
        'lr': float(args.learning_rate)
    }

    LSTM = LSTM(data_path=args.path,
                market_name=args.market,
                parameters=parameters,
                model_path=args.model_save_path,
                model_save_path=args.model_save_path,
                steps=args.step,
                epochs=args.epoch,
                batch_size=args.batch_size,
                gpu=args.gpu,
                fix_init=args.fix_init,
                reload=args.reload
                )
    LSTM.train()











