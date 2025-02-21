from network import NeuralNetwork
from interface import Interface
from customtkinter import CTkLabel
import threading
import random
import numpy as np

class Diane:
    def __init__(self):
        self.network = NeuralNetwork(architecture=[64, 16, 10], learning_rate=0.1)
        self.network.cancel_training = False
        self.training_data, self.input_dim, self.num_classes = self.network.load_training_data('data/training/training_data.json')
        
        self.interface = Interface(self.network, self.training_data, self.input_dim, self.num_classes)
        
        self.interface.mainloop()

if __name__ == '__main__':
    Diane()