from PIL import Image, ImageDraw
import numpy as np
import ImageOperations as iop
import NeuralNetwork as nn
import os
import math
import pickle
import time

image_width = 28
image_height = 28
pool_constant = 7
feature_count = len(os.listdir(iop.FEATURES))

input_node_count = math.ceil(image_width / pool_constant) * math.ceil(image_height / pool_constant) * feature_count
hidden_node_count = image_width * 1
hidden_layer_count = 6
output_node_count = 10

train_amount = 1000
epochs = 50
test_amount = 50

SAVES = 'saves/'

def main():
    neural_network = create_neural_network(input_node_count, hidden_node_count, hidden_layer_count, output_node_count)
    features = iop.read_features()
    
    # train_images, train_labels = read_mnist_data(features, train_amount)
    train_images, train_labels, _ = load_instance(5, train_amount)

    test_images = [iop.normalize_grayscale_image_data(image) for image in 
                    iop.read_mnist_images(iop.MNIST + 't10k-images.idx3-ubyte', test_amount)]
    test_images = process_mnist_images(test_images, features)
    test_labels = iop.read_mnist_labels(iop.MNIST + 't10k-labels.idx1-ubyte', test_amount)
    
    # print(train_images)
    # print(train_labels)
    
    # neural_network.print_network()

    while(control_training(train_images, train_labels, neural_network)):
        for i in range(epochs):
            print('epoch', i)
            train_using_mnist(epochs-i, neural_network, train_images, train_labels)
            accuracy = identify(test_images, test_labels, features, neural_network, test_amount, False)
            print ('epoch', i, "'s accuracy is", accuracy)
            print()
        identify(test_images, test_labels, features, neural_network, test_amount, True)
        # neural_network.print_network()
    
    # image = iop.read_image_data_in_grayscale_normalized(iop.DATA + 'test1.png')
    # iop.save_images_from_normalized_filtered_gsdata(pooled_images)

def control_training(train_images, train_labels, neural_network):
    control_input = input('Continue? (please enter one: stop, save, _empty_) ')
    if control_input == 'stop':
        return False
    elif control_input == 'save':
        save_objects([train_images, train_labels, neural_network])
        return True
    else: 
        return True

def identify(test_images, test_labels, features, neural_network, test_amount, print_mode = True):
    test_out_ids = []
    accuracy = 0
    for i, image in enumerate(test_images):
        if print_mode:
            print('identifying:', i+1)
        test_out_ids.append(identify_character(image, neural_network))
    for i in range(test_amount):
        accuracy += test_out_ids[i] == test_labels[i]
    accuracy /= test_amount
    if print_mode:
        print('Guesses:', test_out_ids)
        print('Actual:', test_labels)
        print('Accuracy:', accuracy)
    return accuracy

def create_neural_network(input_node_count, hidden_node_count, hidden_layer_count, output_node_count):
    return nn.NeuralNetwork(input_node_count, hidden_node_count, hidden_layer_count, output_node_count)

def read_mnist_data(features, quantity):
    train_images_raw = [iop.normalize_grayscale_image_data(image) for image in 
                    iop.read_mnist_images(iop.MNIST + 'train-images.idx3-ubyte', quantity)]
    train_labels = iop.read_mnist_labels(iop.MNIST + 'train-labels.idx1-ubyte', quantity)
    return process_mnist_images(train_images_raw, features), train_labels

def train_using_mnist(epochs, neural_network, train_images, train_labels):
    neural_network.train_using_error_slopes(epochs, train_images, train_labels)
    
def process_mnist_images(images_raw, features):
    train_images = []
    start = 0
    end = 0
    for i, image in enumerate(images_raw):
        if i % 100 == 0:
            start = time.time_ns()
            print('processing image:', i)
        train_images.append(iop.flatten(iop.flatten(iop.filter_and_pool_image(image, features, pool_constant))))
        if i % 100 == 99:
            end = time.time_ns()
            seconds = int(((end - start) / 1e9) * (len(images_raw) - i) / 99)
            minutes = int(seconds / 60)
            seconds %= 60
            print('estimated time remaining:', minutes, 'minutes and', seconds, 'seconds')
    print('processing complete')
    return train_images

def identify_character(inputs, neural_network):
    outputs = neural_network.solve_and_get_outputs(inputs)
    return outputs.index(max(outputs))
    # return outputs
    
def _print_matrix(matrix):
    for i in range(len(matrix)): 
        for j in range(len(matrix[i])): 
            print('{:-6.2}'.format(matrix[i][j]), end = ' ') 
        print() 

def save_objects(obj):
    file_number = len(os.listdir(SAVES)) + 1
    print('saving as save' + str(file_number) + '.ocrnn ...')
    file_to_save = open(SAVES + 'save' + str(file_number) + '.ocrnn', 'wb')
    pickle.dump(obj, file_to_save)
    file_to_save.close()
    print('saving complete')

def load_instance(num, train_amount = -1):
    print('loading data from save' + str(num) + '.ocrnn ...')
    file_to_read = open(SAVES + 'save' + str(num) + '.ocrnn', 'rb')
    data = pickle.load(file_to_read)
    if(train_amount > -1):
        data[0] = data[0][:train_amount]
        data[1] = data[1][:train_amount]
    print('loading complete')
    return data[0], data[1], data[2]

if __name__ == "__main__":
    main()
