# !/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import configparser
import sys
import random
# import win_unicode_console
# win_unicode_console.enable()

# import robot.simple_robot as robot
# import robot.Configuration as config

'''
record:
model size
params
train time
accuracy

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class classifier:

    MODEL_SAVE_PATH = "FULLCNN_model/"
    MODEL_NAME="classify"
    MODEL_4_JAVA = '.\\model4java\\fullconnectedClassify%s_%s.pb'
    MODEL_4_JAVA_TEMP = '.\\model4java\\fullconnectedClassifytemp%s_%s.pb'

    def add_layer(self, inputs, in_size, out_size, activation_function=None, layer_index=0,name=None):
        with tf.variable_scope('layer' + str(layer_index)):
            Weights = tf.Variable(tf.truncated_normal([in_size, out_size], mean=0.0, stddev=0.01, dtype=tf.float32, seed=1), name="weights")
            biases = tf.Variable(tf.zeros([1, out_size], name = "biases") + 0.01)
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases, name = name)
            tf.summary.histogram('histogram_w', Weights)
            tf.summary.histogram('histogram_b', biases)
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
        return outputs

    def accurate_num(self, y_real, y_pred):
        error_statistics = {}
        num = 0
        max_error = 0
        search_length_sum = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_real[i]:
                search_length_sum += 1
                num += 1
            else:
                error = abs(y_pred[i] - y_real[i])
                search_length_sum += error * 2 + 1
                if str(error) in error_statistics.keys():
                    error_statistics[str(error)] += 1
                else:
                    error_statistics[str(error)] = 1
                if error > max_error:
                    max_error = error
        average_search_length = search_length_sum / float(len(y_real))
        return num, max_error, error_statistics, average_search_length


    def shuffle(self, indexs, latitude, longitude):
        randnum = random.randint(0,100)
        random.seed(randnum)
        random.shuffle(indexs)
        random.seed(randnum)
        random.shuffle(latitude)
        random.seed(randnum)
        random.shuffle(longitude)
        return indexs, latitude, longitude

    def load_csv_1(self, fname, label_column, is_shuffle=False):
        indexs = []
        latitude = []
        longitude = []
        with open(fname, "r") as f:
            for line in f:
                cols = line.split(",")
                if len(cols) < 2: continue
                indexs.append([int(cols[label_column].strip())])
                latitude.append([float(cols[0])])
                longitude.append([float(cols[1])])
            # if is_shuffle == True:
            #     return self.shuffle(indexs, latitude, longitude)
            # else:
            return indexs, latitude, longitude

    def load_csv(self, fname, dataset_from, dataset_num, pagesize, label_column):
        i = 1
        indexs = []
        latitude = []
        longitude = []
        values = []
        num = dataset_num + dataset_from
        offset = int(dataset_from / pagesize)
        with open(fname, "r") as f:
            for line in f:
                if i <= dataset_from:
                    i += 1
                    continue
                cols = line.split(",")
                if len(cols) < 2: continue
                indexs.append([int(cols[label_column].strip()) - offset])
                latitude.append([float(cols[0])])
                longitude.append([float(cols[1])])
                values.append([float(cols[0]), float(cols[1])])
                if i < num - 1:
                    i += 1
                else:
                    break
            return indexs, latitude, longitude

    def normalize_data(self, latitude, longitude):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 20))
        latitude = min_max_scaler.fit_transform(latitude)
        longitude = min_max_scaler.fit_transform(longitude)
        return latitude, longitude


    def build_model(self, xs, input_num, output_num, layer_nodes):
        layer_index = 0
        layer = self.add_layer(xs, input_num, layer_nodes[0], activation_function=tf.nn.relu, layer_index=layer_index)
        layer_index += 1
        for i in range(len(layer_nodes) - 1):
            layer = self.add_layer(layer, layer_nodes[i], layer_nodes[i + 1], activation_function=tf.nn.relu, layer_index=layer_index)
            layer_index += 1
        prediction = self.add_layer(layer, layer_nodes[-1], output_num, activation_function=None, layer_index=layer_index, name='prediction')
        return prediction


    def finish_callback(self, fileDir,toUserName):
        friends = robot.search_friends(toUserName)
        for friend in friends:
            friend.send_file(fileDir)

    def get_num_params(self):
        from functools import reduce
        from operator import mul
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        print('参数总数',num_params)
        return num_params

    def deleteModel(self, model_index='', tag=''):
        my_file = self.MODEL_4_JAVA % (tag, model_index)
        if os.path.exists(my_file):
            os.remove(my_file)
        my_file_temp = self.MODEL_4_JAVA_TEMP % (tag, model_index)
        if os.path.exists(my_file_temp):
            os.rename(my_file_temp, my_file)

    def batches(self, features, labels, batch_size=100):
        """
        Create batches of features and labels
        :param batch_size: The batch size
        :param features: List of features
        :param labels: List of labels
        :return: Batches of (Features, Labels)
        """
        assert len(features) == len(labels)
        # TODO: Implement batching
        output_batches = []
        sample_size = len(features)
        for start_i in range(0, sample_size, batch_size):
            end_i = start_i + batch_size
            batch = [features[start_i:end_i],labels[start_i:end_i]]
            output_batches.append(batch)
        return output_batches

    def train_model(self, xs, ys, x_data, y_data, prediction, train_step, loss, num_boost, batch_size, finish_callback=None, fileDir=None, toUserName=None, model_index=0, tag='', output_node_name=''):

        init = tf.global_variables_initializer()
        # xs = tf.placeholder(tf.float32, [None, 2])
        # ys = tf.placeholder(tf.float32, [None, 1])

        # correct_prediction = tf.equal(tf.to_int32(y_data, name='ToInt32'), tf.to_int32(prediction, name='ToInt32'))
        # correct_prediction = tf.equal(tf.to_int32(y_data, name='ToInt32'), tf.to_int32(tf.reduce_max(prediction, reduction_indices=[1]), name='ToInt32'))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log", tf.get_default_graph())
        saver = tf.train.Saver()
        num_params = 0
        max_score = 0
        print('prediction:',prediction)
        with tf.Session() as sess:
            sess.run(init)
            num_params = self.get_num_params()
            max_score_index = 0
            min_average_search_length = 0
            max_search_length = 0
            min_loss = 1000
            min_loss_index = 10
            average_search_length_result = 0
            max_error = 0
            for i in range(num_boost):
                # training
                if batch_size == 0:
                    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
                else:
                    for batch_features, batch_labels in self.batches(x_data, y_data, batch_size):
                        sess.run(train_step, feed_dict={xs: batch_features, ys: batch_labels})
                #
                if i % 1000 == 0:
                    # 配置运行时需要记录的信息。
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
                    # 运行时记录运行信息的proto。
                    run_metadata = tf.RunMetadata()
                    prediction_value = sess.run(prediction, feed_dict={xs: x_data})
                    score = accuracy_score(y_data, prediction_value.argmax(axis=1))
                    num, max_error_temp, error_statistics, average_search_length = self.accurate_num(y_data,
                                                                                           prediction_value.argmax(axis=1))
                    summary,loss_value = sess.run([merged,loss], feed_dict={xs: x_data, ys: y_data}, run_metadata=run_metadata, options=run_options)
                    writer.add_run_metadata(run_metadata=run_metadata, tag=("tag%d" % i), global_step=i)
                    writer.add_summary(summary, i)
                    # saver.save(sess, os.path.join(self.MODEL_SAVE_PATH, self.MODEL_NAME))
                    # print('score:', score, ' average_search_length,', average_search_length)
                    print('score:'+str(score))
                    if max_score < score:
                        max_score = score
                        max_score_index = i
                        average_search_length_result = average_search_length
                        max_error = max_error_temp
                        # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['layer1/prediction'])
                        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=[output_node_name])
                        with tf.gfile.FastGFile(self.MODEL_4_JAVA_TEMP % (tag, model_index), mode='w') as f:
                            f.write(output_graph_def.SerializeToString())
                    if i >= 3000 and score*len(y_data) == 100:
                        break

                else:
                    summary,loss_value = sess.run([merged,loss], feed_dict={xs: x_data, ys: y_data})
                    # if min_loss > loss_value:
                    #     min_loss = loss_value
                    #     min_loss_index = i
            # print('max_score:%.4f, max_score_index:%f, min_loss:%.4f, min_loss_index:%f, average_search_length:%f, max_error:%f' % (
            # max_score, max_score_index, min_loss, min_loss_index, average_search_length_result, max_error))
            # if finish_callback is not None:
            #     self.finish_callback(fileDir, toUserName)

            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=[output_node_name])
        # with tf.gfile.FastGFile('.\\model4java\\fullconnectedClassify' + str(model_index) + '.pb', mode='w') as f:
        #     f.write(output_graph_def.SerializeToString())
            # file = open('../logs/classify/level_' + str(label_column - 1)+ '_.txt','a')
            # file.write(str(max_score) + ',' + str(average_search_length_result[0]) + ',' + str(max_error[0])+'\r\n')
            # file.close()
            # return max_score,average_search_length_result,max_error
        self.deleteModel(model_index, tag)
        return max_score, num_params, average_search_length_result

    def loss_func(self, prediction, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def test_model(self, xs, x_data, y_data, prediction):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # print(prediction_value.argmax(axis=1))
            score = accuracy_score(y_data, prediction_value.argmax(axis=1))
            num, max_error_temp, error_statistics, average_search_length = self.accurate_num(y_data, prediction_value.argmax(axis=1))
            # print('accuracy score:', score)
            # print('num :', num)
            # print('max_error_temp :', max_error_temp)
            # print('average_search_length:', average_search_length)
        return score, average_search_length

    def change_output(self, result_file):
        temp = sys.stdout
        file = open(result_file,'a')
        sys.stdout = file

    def init_robot(self):
        configuration = config.Configuration()
        configuration.set_receiver('刘冠利')
        configuration.read()
        myrobot = robot.init_robot()
        return configuration,myrobot

    def find_best_model(self, xs, dataset_num, label_column, prediction, test_dataset_list, dataset_template=None):
        average_search_length = 0.0
        average_score = 0.0
        # for i in range(100):
        # for i in test_dataset_list:
        #     test_dataset_file = dataset_template % (i)
        #     # print('test_dataset_file',test_dataset_file)
        #     # test_dataset_file = "../dataset/random_10000_" + str(i) + "_zvalue_100_.csv"
        #     # test_dataset_file = config.get('FullConnectedNetwork', 'test_dataset')
        #     # y_data, latitude, longitude = load_csv(test_dataset_file, dataset_from, dataset_num, page_size, label_column)
        #     y_data, latitude, longitude = self.load_csv_1(test_dataset_file, label_column)
        #     # print(indexs)
        #     latitude, longitude = self.normalize_data(latitude, longitude)
        #     x_data = np.hstack((latitude, longitude))
        #     score, search_length = self.test_model(xs, x_data, y_data, prediction)
        #     average_search_length += search_length
        #     average_score += score
        # #TODO 这里的长度计算可以优化一下
        # average_search_length = average_search_length/len(test_dataset_list)
        # average_score = average_score/len(test_dataset_list)
        # print('-------------------------------------------------------')
        # print('accuracy score:', average_score)
        # print('average_search_length:', average_search_length)
        return average_score, average_search_length

    def check_model(self, model_index='', tag=''):
        my_file = self.MODEL_4_JAVA % (tag, model_index)
        if os.path.exists(my_file):
            return True
        else:
            return False

    def main(self, model_index='0'):
        # configuration,myrobot = init_robot()
        xs = tf.placeholder(tf.float32, [None, 2],name='x-input')
        ys = tf.placeholder(tf.int32, [None, 1],name='y-input')
        config = configparser.RawConfigParser()
        config.read('experiments.cfg')
        dataset_file = config.get('FullConnectedNetwork', 'dataset')
        learning_rate = config.getfloat('FullConnectedNetwork', 'learning_rate')
        epoch_size = config.getint('FullConnectedNetwork', 'epoch_size')
        dataset_num = config.getint('FullConnectedNetwork', 'dataset_num')
        dataset_from = config.getint('FullConnectedNetwork', 'dataset_from')
        label_column = config.getint('FullConnectedNetwork', 'label_column')
        # page_size = config.getint('FullConnectedNetwork', 'page_size')
        dataset_template = config.get('FullConnectedNetwork', 'dataset_template')
        output_node_name = config.get('FullConnectedNetwork', 'output_node_name')
        # class_num = config.getint('FullConnectedNetwork', 'class_num')
        hidden_layers = config.get('FullConnectedNetwork', 'hidden_layers')
        result_file = '../logs/classify/dataset_' + dataset_file.split('/')[-1].split('.')[0] + '_' + hidden_layers + '.txt'
        is_output_to_file = config.getboolean('FullConnectedNetwork', 'is_output_to_file')
        test_dataset = config.get('FullConnectedNetwork', 'test_dataset')
        tag = config.get('FullConnectedNetwork', 'tag')
        batch_size = config.getint('FullConnectedNetwork', 'batch_size')
        is_shuffle = config.getboolean('FullConnectedNetwork', 'is_shuffle')
        if is_output_to_file:
            change_output(result_file)

        print('-------------------------------------------------------')

        max_average_score = 0
        max_average_score_index = 0
        min_average_search_length = 2
        min_average_search_length_index = 0

        # for i in range(100):
            # dataset_file = "../dataset/random_10000_" + str(i) + "_zvalue_80_.csv"
        hidden_layers_list = list(map(int, hidden_layers.split(',')))
        # if test_dataset == "":
        #     test_dataset_list = [model_index]
        # else:
        #     test_dataset_list = list(map(int, test_dataset.split(',')))
        # indexs, latitude, longitude = load_csv(dataset_file, dataset_from, dataset_num, page_size, label_column)

        indexs, latitude, longitude = self.load_csv_1(dataset_file, label_column, is_shuffle)
        dataset_num = len(indexs)
        # print(indexs)
        latitude, longitude = self.normalize_data(latitude, longitude)
        x_data = np.hstack((latitude, longitude))
        y_data = indexs
        print(len(self.batches(x_data, y_data)))
        # for batch_features, batch_labels in self.batches(x_data, y_data):
        #     print(batch_features)
        #     print(batch_labels)
        print(indexs[-1][0])
        class_num = indexs[-1][0] + 1
        # if class_num < 100:
        #     return
        labels = np.array(y_data).reshape(len(y_data))
        prediction = self.build_model(xs, 2, class_num, hidden_layers_list)
        ys_ = tf.reshape(ys, [-1])
        loss = self.loss_func(prediction, ys_)
        # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        train_step = tf.train.AdagradOptimizer(0.5).minimize(loss)
        # train_step = tf.train.AdadeltaOptimizer(0.5).minimize(loss)
        # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        # train_step = tf.train.MomentumOptimizer(0.5, 0.9).minimize(loss)
        # train_model(x_data, y_data, prediction, train_step, loss, epoch_size, finish_callback, result_file, configuration.receiver)
        max_score, num_params, average_search_length = self.train_model(xs, ys, x_data, y_data, prediction, train_step, loss, epoch_size, batch_size, model_index=model_index, tag=tag, output_node_name=output_node_name)
        print('--------------------------' + 'train finish' + '-----------------------------')
        # average_score, average_search_length = self.find_best_model(xs, dataset_num, label_column, prediction, test_dataset_list, dataset_template)
        # 一定要每次清除图上的数据
        tf.reset_default_graph()
        return max_score, num_params, average_search_length
        # return average_search_length
            # if max_average_score < average_score:
            #     max_average_score = average_score
            #     max_average_score_index = i
            # if min_average_search_length > average_search_length:
            #     min_average_search_length = average_search_length
            #     min_average_search_length_index = i
        # print('-------------------------------------------------------')
        # print('max_average_score',max_average_score)
        # print('max_average_score_index',max_average_score_index)
        # print('min_average_search_length',min_average_search_length)
        # print('min_average_search_length_index',min_average_search_length_index)
        # print('-------------------------------------------------------')
