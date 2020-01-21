import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
import random

from genetic_algorithm.individual import Individual
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name
from utils import SMB, StaticTileType
from config import Config


class Mario(Individual):
    def __init__(self,
                 config: Config,
                 chromosome: Optional[Dict[str, np.ndarray]] = None,
                 hidden_layer_architecture: List[int] = [12, 9],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 lifespan: Union[int, float] = np.inf,
                 ):
        
        self.lifespan = lifespan
        self._fitness = 0  # Overall fitness
        self._frames_since_progress = 0  # Number of frames since Mario has made progress towards the goal
        self._frames = 0  # Number of frames Mario has been alive
        
        self.hidden_layer_architecture = hidden_layer_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.config = config

        #@TODO: Set this based off how the input is (2,2,2,2), etc
        u, d, l, r = self.config.NeuralNetwork.inputs_size
        self.u, self.d, self.l, self.r = u, d, l, r
        ud = int(bool(u and d))  # If both u and d directions are non-zero, there is an additional square (Mario)
        lr = int(bool(l and r))  # If both l and r directions are non-zero, there is an additional square (Mario)
        num_inputs = (u + d + ud) * (l + r + lr)
        
        self.inputs_as_array = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # Input Nodes
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden Layer Ndoes
        self.network_architecture.append(6)                               # 6 Outputs ['u', 'd', 'l', 'r', 'a', 'b']

        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation)
                                         )

        # If chromosome is set, take it
        if chromosome:
            self.network.params = chromosome
        
        self.is_alive = True

    @property
    def fitness(self):
        return self._fitness

    @property
    def chromosome(self):
        pass

    def decode_chromosome(self):
        pass

    def encode_chromosome(self):
        pass

    def set_input_as_array(self, ram, tiles) -> None:
        mario_row, mario_col = SMB.get_mario_row_col(ram)
        arr = []
        #@TODO: Where did I mess up the row/col
        for row in range(-self.l, self.r+1):
            for col in range(-self.u, self.d+1):
                try:
                    t = tiles[(row + mario_row, col + mario_col)]
                except:
                    t = StaticTileType(0x00)
                print('{:02X} '.format(t.value), end = '')
                arr.append(t.value)
            print()
        print(arr)
        
        print()
        self.inputs_as_array = np.array(arr).reshape((-1,1)) 

    def update(self, ram) -> bool:
        """
        The main update call for Mario.
        Takes in inputs of surrounding area and feeds through the Neural Network
        
        Return: True if Mario is alive
                False otherwise
        """
        if self.is_alive:
            self._frames += 1

        if ram[0x0E] in (0x0B, 0x06):
            self.is_alive = False

        return self.is_alive
