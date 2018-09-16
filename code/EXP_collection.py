#!/usr/bin/python
#coding=utf-8

import configparser
import os
import sys
import time
from FullConnectedNetWork_Classify_class import classifier

config = configparser.RawConfigParser()
# dataset_template = '../dataset/random_10000_%d_zvalue_100_.csv'
training_data = "D:\\UniMelbourne\\DL_index4R_tree\\dataset\\"
dataset_template = 'D:\\UniMelbourne\\DL_index4R_tree\\dataset\\kdb_tree\\training\\%d.csv'
log = ".\\training_time\\"
# dataset_template = 'D:\\UniMelbourne\\DL_index4R_tree\\dataset\\real_dataset\\dataset_west_13m_10000_%d_zvalue_100_.csv'
# dataset_template = 'D:\\UniMelbourne\\DL_index4R_tree\\dataset\\uniform\\1M\\random_10000_%d_zvalue_100_.csv'
# dataset_template = 'D:\\UniMelbourne\\dataset4experiment\\normal\\8M\\random_tf_%d_zvalue.csv'
# dataset_template = 'D:\\UniMelbourne\\DL_index4R_tree\\dataset\\skewed\\compounded\\random_10000_%d_zvalue_100_.csv'
tag = ''
# When adding sections or items, add them in the reverse order of
# how you want them to be displayed in the actual file.
# In addition, please note that using RawConfigParser's and the raw
# mode of ConfigParser's respective set functions, you can assign
# non-string values to keys internally, but will receive an error
# when attempting to write to a file or when you get it in non-raw
# mode. SafeConfigParser does not allow such assignments to take place.
def init_config():
    config.add_section('FullConnectedNetwork')
    config.set('FullConnectedNetwork', 'dataset', '../dataset/random_10000_0_zvalue_100_.csv')
    config.set('FullConnectedNetwork', 'dataset_num', '10000')
    config.set('FullConnectedNetwork', 'dataset_from', '1')
    config.set('FullConnectedNetwork', 'label_column', '2')
    config.set('FullConnectedNetwork', 'n_layer', '4')
    config.set('FullConnectedNetwork', 'learning_rate', '0.01')
    config.set('FullConnectedNetwork', 'epoch_size', '40000')
    config.set('FullConnectedNetwork', 'class_num', '100')
    # config.set('FullConnectedNetwork', 'page_size', '100')
    config.set('FullConnectedNetwork', 'hidden_layers', '80,80,80')
    config.set('FullConnectedNetwork', 'is_output_to_file', 'False')
    config.set('FullConnectedNetwork', 'dataset_template', dataset_template)
    config.set('FullConnectedNetwork', 'tag', '')
    config.set('FullConnectedNetwork', 'output_node_name', 'layer3/prediction')
    config.set('FullConnectedNetwork', 'test_dataset', '../dataset/random_10000_1_zvalue_100_.csv')
    config.set('FullConnectedNetwork', 'is_shuffle', True)


def write_file():
    # Writing our configuration file to 'example.cfg'
    with open('experiments.cfg', 'w') as configfile:
        config.write(configfile)

def load_csv_1(fname, label_column):
    i = 1
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
        return indexs, latitude, longitude

def execute_each_block_model():
    for i in range(50):
        # if i < 100:
        #     continue
        dataset_file = dataset_template % (i)
        y_data, latitude, longitude = load_csv_1(dataset_file, 2)
        key = int(y_data[-1][0] / 10)
        hidden_layers = '80,80,80'
        # if key <= 1:
        #     hidden_layers = '5,5'
        # elif key < 5:
        #     hidden_layers = [str((key - 1) * 10),str((key - 1) * 10)]
        #     hidden_layers = ",".join(hidden_layers)
        # elif key < 10:
        #     hidden_layers = '80,80'
        # elif key < 15:
        #     hidden_layers = '120,120'
        # else:
        #     hidden_layers = '200,200'
        config.set('FullConnectedNetwork', 'hidden_layers', hidden_layers)
        config.set('FullConnectedNetwork', 'tag', tag)
        config.set('FullConnectedNetwork', 'test_dataset', "")
        config.set('FullConnectedNetwork', 'dataset', dataset_file)
        write_file()
        model = classifier()
        average_search_length = model.main(i)

def execute_each_block_model_epoch_size(hidden_layers, output_node_name, layer_tag):
    for i in range(40):
        # if i < 100:
        #     continue
        dataset_file = dataset_template % (i % 10)
        y_data, latitude, longitude = load_csv_1(dataset_file, 2)
        key = int(y_data[-1][0] / 10)
        # if key <= 1:
        #     hidden_layers = '5,5'
        # elif key < 5:
        #     hidden_layers = [str((key - 1) * 10),str((key - 1) * 10)]
        #     hidden_layers = ",".join(hidden_layers)
        # elif key < 10:
        #     hidden_layers = '80,80'
        # elif key < 15:
        #     hidden_layers = '120,120'
        # else:
        #     hidden_layers = '200,200'
        epoch_size = str(10000 * (int(i/10) +1))
        tag = "uniform_" + epoch_size + '_' + layer_tag
        config.set('FullConnectedNetwork', 'epoch_size', epoch_size)
        config.set('FullConnectedNetwork', 'hidden_layers', hidden_layers)
        config.set('FullConnectedNetwork', 'tag', tag)
        config.set('FullConnectedNetwork', 'test_dataset', "")
        config.set('FullConnectedNetwork', 'dataset', dataset_file)
        config.set('FullConnectedNetwork', 'output_node_name', output_node_name)
        write_file()
        model = classifier()
        average_search_length = model.main(i)

def execute_each_block_model_layer_nodes_():
    config.set('FullConnectedNetwork', 'output_node_name', 'layer1/prediction')
    write_file()
    execute_each_block_model_layer_nodes('60', 60)
    execute_each_block_model_layer_nodes('100', 100)
    execute_each_block_model_layer_nodes('120', 120)
    config.set('FullConnectedNetwork', 'output_node_name', 'layer2/prediction')
    write_file()
    execute_each_block_model_layer_nodes('60,60', 60)
    execute_each_block_model_layer_nodes('100,100', 100)
    execute_each_block_model_layer_nodes('120,120', 120)
    config.set('FullConnectedNetwork', 'output_node_name', 'layer3/prediction')
    write_file()
    execute_each_block_model_layer_nodes('60,60,60', 60)
    execute_each_block_model_layer_nodes('100,100,100', 100)
    execute_each_block_model_layer_nodes('120,120,120', 120)


def execute_each_block_model_layer_nodes(hidden_layers, num):
    for i in range(10):
        # if i < 100:
        #     continue
        dataset_file = dataset_template % i
        y_data, latitude, longitude = load_csv_1(dataset_file, 2)
        key = int(y_data[-1][0] / 10)
        # hidden_layers = '80,80,80'
        # if key <= 1:
        #     hidden_layers = '5,5'
        # elif key < 5:
        #     hidden_layers = [str((key - 1) * 10),str((key - 1) * 10)]
        #     hidden_layers = ",".join(hidden_layers)
        # elif key < 10:
        #     hidden_layers = '80,80'
        # elif key < 15:
        #     hidden_layers = '120,120'
        # else:
        #     hidden_layers = '200,200'
        tag = "uniform_" + str(num) + "_" + hidden_layers
        config.set('FullConnectedNetwork', 'epoch_size', 40000)
        config.set('FullConnectedNetwork', 'hidden_layers', hidden_layers)
        config.set('FullConnectedNetwork', 'tag', tag)
        config.set('FullConnectedNetwork', 'test_dataset', "")
        config.set('FullConnectedNetwork', 'dataset', dataset_file)
        write_file()
        model = classifier()
        average_search_length = model.main(i)

def dataset_analysis():
    dataset_partition={}

    length = len(os.listdir(training_data))
    for i in range(length):
        # test_dataset_file = "../dataset/random_skewed_" + str(i) + "_zvalue_100_.csv"
        # test_dataset_file = "../dataset/random_skewed_" + str(i) + "_zvalue_100_.csv"
        test_dataset_file = dataset_template % (i)
        # test_dataset_file = config.get('FullConnectedNetwork', 'test_dataset')
        # y_data, latitude, longitude = load_csv(test_dataset_file, dataset_from, dataset_num, page_size, label_column)
        y_data, latitude, longitude = load_csv_1(test_dataset_file, 2)
        # print('dataset size ', len(y_data))
        # print('class num ', y_data[-1][0])
        class_num = int(y_data[-1][0] / 10)
        key = class_num
        if key in dataset_partition.keys():
            dataset_partition[key].append(str(i))
        else:
            dataset_partition[key] = [str(i)]
    return dataset_partition

# dataset_file = "../dataset/random_10000_" + str(19) + "_zvalue_80_.csv"
def find_max(item):

    file = log + tag + '_' + item + '.txt'

    with open(file, 'w+') as f:
        time_start=time.time()
        dataset = dataset_template % (int(item))
        print('--------------------------dataset:' +dataset + '-----------------------------')
        max_score = 0
        score=0
        max_score_width = 0
        width = 3
        while width < 11:
            hidden_layers = [str(width * 10),str(width * 10),str(width * 10)]
            hidden_layers = ",".join(hidden_layers)
            width += 1
            config.set('FullConnectedNetwork', 'hidden_layers', hidden_layers)
            #     hidden_layers = ",".join(hidden_layers)
            config.set('FullConnectedNetwork', 'dataset', dataset)
            write_file()
            model = classifier()
            score = model.main(int(item))
            if max_score < score:
                max_score = score
                max_score_width = width
        time_end=time.time()


def execute():
    dataset_partition = dataset_analysis()
    for index, key in enumerate(dataset_partition.keys()):
        dataset_list = dataset_partition[key]
        print("key:",key, " number of related datasets",len(dataset_list))
        hidden_layers = '30,30,30'
        config.set('FullConnectedNetwork', 'tag', tag)
        config.set('FullConnectedNetwork', 'test_dataset', ",".join(dataset_list))
        print('--------------------------key:' + str(key) + '-----------------------------')
        for item in dataset_list:
            find_max(item)
        print('--------------------------最终的数据集:' +item + '-----------------------------')

def trainAll():
    list = os.listdir(training_data) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
        path = os.path.join(training_data,list[i])
        # algorithm_type = path.split('_')[0]
        # train(path, 3, 2000)
        # train(path, 3, 200)
        train(path, 3, 0)
        # train_trapezoid_1(path, [str(40),str(60),str(80)], '468')
        # train_trapezoid_1(path, [str(80),str(60),str(40)], '864')
        # train_trapezoid_1(path, [str(80),str(100),str(120)], '81012')
        # train_trapezoid_1(path, [str(60),str(80),str(100)], '6810')
        # train_trapezoid_1(path, [str(100),str(120),str(140)], '101214')
        # train_trapezoid_1(path, [str(120),str(140),str(160)], '121416')

# hidden_layers = [str(40),str(60),str(80)]
def train_trapezoid_1(training_set, hidden_layers,layer_tag):
    batch_size = 0
    names = training_set.split('\\')
    model_index = names[len(names) - 1].split('.')[0]
    file = log + model_index + str(batch_size) + '_' + layer_tag + '_.txt'
    with open(file, 'w+') as f:
        dataset = training_set
        print('--------------------------dataset:' +dataset + '-----------------------------')
        score=0
        width = 2
        num_params = 0
        time_start=time.time()

        hidden_layers = ",".join(hidden_layers)
        layer_num = 3
        config.set('FullConnectedNetwork', 'output_node_name', 'layer' + str(layer_num) + '/prediction')
        config.set('FullConnectedNetwork', 'hidden_layers', hidden_layers)
        config.set('FullConnectedNetwork', 'dataset', dataset)
        config.set('FullConnectedNetwork', 'batch_size', batch_size)
        tag = '_width_' + str(hidden_layers) + '_layernum_' + str(layer_num) + "_batch_size_" + str(batch_size)
        config.set('FullConnectedNetwork', 'tag', tag)
        print('-------------------------')
        print('--------width--------:' + str(width))
        print('--------layer_num--------:' + str(layer_num))

        write_file()
        model = classifier()
        if model.check_model(model_index, tag):
            print('jixu')
        score, num_params, average_search_length = model.main(model_index)
        time_end=time.time()
        f.write('-------------------------\n')
        f.write('--------score--------:' + str(score) + '\n')
        f.write('--------average_search_length--------:' + str(average_search_length) + '\n')
        f.write('--------time--------:' + str(time_end - time_start) + 's' + '\n')
        f.write('--------num_params--------:' + str(num_params) + '\n')

def train(training_set, layer_num_max, batch_size):
    names = training_set.split('\\')
    model_index = names[len(names) - 1].split('.')[0]
    file = log + model_index + str(batch_size) + '_.txt'
    with open(file, 'a+') as f:
        for layer_num in range(layer_num_max):
            layer_num = layer_num + 1
            dataset = training_set
            print('--------------------------dataset:' +dataset + '-----------------------------')
            max_score = 0
            score=0
            max_score_width = 0
            width = 10
            num_params = 0
            while width < 31:
                time_start=time.time()
                width += 2
                hidden_layers = []
                for i in range(layer_num):
                    hidden_layers.append(str(width * 10))
                # hidden_layers = [str(width * 10),str(width * 10),str(width * 10)]
                hidden_layers = ",".join(hidden_layers)
                config.set('FullConnectedNetwork', 'output_node_name', 'layer' + str(layer_num) + '/prediction')
                config.set('FullConnectedNetwork', 'hidden_layers', hidden_layers)
                config.set('FullConnectedNetwork', 'dataset', dataset)
                config.set('FullConnectedNetwork', 'batch_size', batch_size)
                tag = '_width_' + str(width) + '_layernum_' + str(layer_num) + "_batch_size_" + str(batch_size)
                config.set('FullConnectedNetwork', 'tag', tag)
                print('-------------------------')
                print('--------width--------:' + str(width))
                print('--------layer_num--------:' + str(layer_num))

                write_file()
                model = classifier()
                if model.check_model(model_index, tag):
                    print('jixu')
                    continue
                score, num_params, average_search_length = model.main(model_index)
                if max_score < score:
                    max_score = score
                    max_score_width = width

                time_end=time.time()

            # print('--------max_score--------:' + str(max_score))
            # print('--------time--------:' + str(time_end - time_start) + 's')
            # print('--------width--------:' + str(max_score_width * 10) + 's')
            # print('--------num_params--------:' + str(num_params) + 's')
                f.write('-------------------------\n')
                f.write('--------width--------:' + str(width * 10) + '\n')
                f.write('--------layer_num--------:' + str(layer_num) + '\n')
                f.write('--------score--------:' + str(score) + '\n')
                f.write('--------average_search_length--------:' + str(average_search_length) + '\n')
                f.write('--------time--------:' + str(time_end - time_start) + 's' + '\n')
                f.write('--------num_params--------:' + str(num_params) + '\n')


def main():
    init_config()
    trainAll()
    # execute()
    # execute_each_block_model()
    # execute_each_block_model_epoch_size('80', 'layer1/prediction', 'layer1')
    # execute_each_block_model_epoch_size('80,80', 'layer2/prediction', 'layer2')
    # execute_each_block_model_epoch_size('80,80,80', 'layer3/prediction', 'layer3')
    # execute_each_block_model_layer_nodes_()

if __name__ == '__main__':
    main()
