import random as rnd
import numpy as np
import math
import time

class Node:
    
    def __init__(self, in_weights_count):
        self.value = 0
        self.bias = 0
        self.in_weights = []
        self.error_delta = []
        self.generate_random_weights_and_bias(in_weights_count)

    def generate_random_weights_and_bias(self, in_weights_count):
        for i in range(in_weights_count):
            # self.in_weights.append(rnd.random())
            self.in_weights.append(rnd.random() * 2 - 1)
        self.bias = 0
        # self.bias = rnd.random()
        # self.bias = rnd.random() * 2 - 1

    def print_node(self):
        print('value: ', self.value, end=' ')
        print('weights:', self.in_weights)
        
    def normalize_value(self):
        # self.value = (1 / (1 + np.exp(-self.value)))
        self.value = (2 / (1 + np.exp(-self.value))) - 1

#let's first do a basic/standard network with one type of node and layer
class Layer:
    
    def __init__(self, layer_size, in_weights):
        self.layer_size = int(layer_size)
        self.nodes = [Node(in_weights) for i in range(self.layer_size)]
        
    def print_layer(self):
        for node in self.nodes:
            node.print_node()
            
class NeuralNetwork:
    
    # ------------------------------neural creator-------------------------------------    
 
    def __init__(self, input_count, hidden_count, hidden_size, output_count):
        self.layer_count = hidden_size + 2
        self.layers = []
        self.create_layers(input_count, hidden_count, hidden_size, output_count)
        
    def create_layers(self, input_count, hidden_count, hidden_size, output_count):
        self.layers.append(Layer(input_count, 0))
        for i in range(hidden_size):
            self.layers.append(Layer(hidden_count, self.layers[len(self.layers) - 1].layer_size))
        self.layers.append(Layer(output_count, hidden_count))
        
    def get_network_data(self):
        # network[layer][node -> (value, [in_weights])]
        return [[(node.value, node.in_weights) for node in layer] for layer in self.layers]
    
    def print_network(self):
        for layer in self.layers:
            layer.print_layer()
            print()
    
    # ------------------------------neural solver-------------------------------------    
    
    def solve_and_get_outputs(self, inputs):
        self.solve(inputs)
        return [node.value for node in self.layers[self.layer_count - 1].nodes]
    
    def solve(self, inputs):
        self.assign_inputs(inputs)
        self.calculate();
    
    def assign_inputs(self, inputs):
        i = 0
        for i in range(len(inputs)):
            self.layers[0].nodes[i].value = inputs[i]
            i+=1
    
    def calculate(self): 
        l = 0
        for layer in self.layers:
            if l == 0:
                l+=1
                continue
            for node in layer.nodes:
                node.value = 0
                for i in range(len(node.in_weights)):
                    node.value += self.layers[l-1].nodes[i].value * node.in_weights[i]
                node.value += node.bias
                node.normalize_value();
            l+=1
    
    # --------------------------------error slopes------------------------------------
    
    def train_using_error_slopes(self, epochs, training_data, training_label):
        # expected = [0] * self.layers[self.layer_count - 1].layer_size
        expected = [-1] * self.layers[self.layer_count - 1].layer_size
        start = 0
        end = 0
        for i, trial_inputs in enumerate(training_data):
            if i % 100 == 0:
                start = time.time_ns()
                print('training:', i)
            if i > 0:
                # expected[training_label[i-1]] = 0  # update expected values
                expected[training_label[i-1]] = -1  # update expected values
            expected[training_label[i]] = 1
            self._forward_propagate_inputs(trial_inputs)
            self._back_propagate_errors(expected)
            self._update_weights(trial_inputs)
            if i % 100 == 99:
                end = time.time_ns()
                seconds = int(((end - start) / 1e9) * ((epochs * len(training_data) - i) / 100))
                minutes = int(seconds / 60)
                seconds %= 60
                print('estimated time remaining:', minutes, 'minutes and', seconds, 'seconds')
    
    def _forward_propagate_inputs(self, inputs):
        self.solve(inputs)
    
    # def _back_propagate_errors(self, expected):
    #    for layer_idx, layer in reversed(list(enumerate(self.layers))):
    #        errors = []
    #        if(layer_idx == self.layer_count - 1): # output layer
    #            for node_idx in range(layer.layer_size):
    #                errors.append(expected[node_idx] - layer.nodes[node_idx].value)
    #        else:
    #            next_layer = self.layers[layer_idx + 1]
    #            for node_idx in range(layer.layer_size):
    #                error = []
    #                for next_node in next_layer.nodes:
    #                    error.append(next_node.in_weights[node_idx] * next_node.error_delta)
    #                errors.append(error)
    #        for node_idx, node in enumerate(layer.nodes):
    #            node.error_delta = errors[node_idx] * self._sigmoid_transfer_derivative(node.value) 
    
    def _back_propagate_errors(self, expected):
        for layer_idx, layer in reversed(list(enumerate(self.layers))):
            errors = []
            if(layer_idx == self.layer_count - 1): # output layer
                for node_idx in range(layer.layer_size):
                    errors.append(expected[node_idx] - layer.nodes[node_idx].value)
            else:
                next_layer = self.layers[layer_idx + 1]
                for node_idx in range(layer.layer_size):
                    error = 0
                    for next_node in next_layer.nodes:
                        error += next_node.in_weights[node_idx] * next_node.error_delta
                    errors.append(error)
            for node_idx, node in enumerate(layer.nodes):
                node.error_delta = errors[node_idx] * self._sigmoid_transfer_derivative(node.value) 
    
    def _sigmoid_transfer_derivative(self, output):
        return output * (1 - output)
        # return 2 * output * (1 - output)
        
    # def _update_weights(self, first_inputs, learning_rate = 0.01):
    #     for layer_idx in range(self.layer_count):
    #         if layer_idx == 0:
    #             inputs = first_inputs
    #         else:
    #             inputs = [node.value for node in self.layers[layer_idx - 1].nodes]
    #         for node_idx, node in enumerate(self.layers[layer_idx].nodes):
    #             for wi_idx, weight in enumerate(node.in_weights):
    #                 node.in_weights[wi_idx] += node.error_delta * learning_rate * inputs[wi_idx]
    #                 # print(node.error_delta * learning_rate * inputs[wi_idx])
    #             node.bias += node.error_delta * learning_rate
    
    def _update_weights(self, first_inputs, learning_rate = 0.00005):
        for layer_idx in range(self.layer_count):
            if layer_idx == 0:
                inputs = first_inputs
            else:
                inputs = [node.value for node in self.layers[layer_idx - 1].nodes]
            for node_idx, node in enumerate(self.layers[layer_idx].nodes):
                for wi_idx, weight in enumerate(node.in_weights):
                    node.in_weights[wi_idx] += node.error_delta * learning_rate * inputs[wi_idx]
                    # print(node.error_delta * learning_rate * inputs[wi_idx])
                node.bias += node.error_delta * learning_rate
        
                
    
    