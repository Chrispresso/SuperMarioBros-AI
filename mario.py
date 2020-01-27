import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
import random
import os

from genetic_algorithm.individual import Individual
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name
from utils import SMB, StaticTileType, EnemyType
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

        u, d, l, r = self.config.NeuralNetwork.inputs_size
        self.u, self.d, self.l, self.r = u, d, l, r
        ud = int(bool(u and d))  # If both u and d directions are non-zero, there is an additional square (Mario)
        lr = int(bool(l and r))  # If both l and r directions are non-zero, there is an additional square (Mario)
        num_inputs = (u + d + ud) * (l + r + lr)
        
        self.inputs_as_array = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # Input Nodes
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden Layer Ndoes
        self.network_architecture.append(6)                        # 6 Outputs ['u', 'd', 'l', 'r', 'a', 'b']

        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation)
                                         )

        # If chromosome is set, take it
        if chromosome:
            self.network.params = chromosome
        
        self.is_alive = True
        self.x_dist = None
        self.game_score = None

        # Keys correspond with             B, NULL, SELECT, START, U, D, L, R, A
        # index                            0  1     2       3      4  5  6  7  8
        self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)
        self.farthest_x = 0


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

    def calculate_fitness(self):
        frames = self._frames
        distance = self.x_dist
        score = self.game_score

        self._fitness = self.config.GeneticAlgorithm.fitness_func(frames, distance, score)

    def set_input_as_array(self, ram, tiles) -> None:
        mario_row, mario_col = SMB.get_mario_row_col(ram)
        print(mario_row, mario_col)
        arr = []
        #@TODO: Where did I mess up the row/col
        # for col in range(-self.l, self.r+1):
        #     for row in range(-self.u, self.d+1):
        for row in range(-self.u, self.d+1):
            for col in range(-self.l, self.r+1):
                try:
                    t = tiles[(row + mario_row, col + mario_col)]
                    if isinstance(t, StaticTileType):
                        if t.value == 0:
                            arr.append(0)
                        else:
                            arr.append(1)
                    elif isinstance(t, EnemyType):
                        arr.append(-1)
                    else:
                        raise Exception("wit") #@TODO?
                except:
                    t = StaticTileType(0x00)
                    arr.append(0) # Empty
                
                print('{:02} '.format(arr[-1]), end = '')
            print()
        # print(arr)
        
        print()
        # print(ram)
        # import sys
        # sys.exit(-1)
        # SMB.get_tiles(ram, q=False)
        self.inputs_as_array = np.array(arr).reshape((-1,1)) 
        print(', '.join([str(x[0]) for x in self.inputs_as_array]))


    def update(self, ram, tiles, buttons, ouput_to_buttons_map) -> bool:
        """
        The main update call for Mario.
        Takes in inputs of surrounding area and feeds through the Neural Network
        
        Return: True if Mario is alive
                False otherwise
        """
        if self.is_alive:
            self._frames += 1
            self.x_dist = SMB.get_mario_location_in_level(ram).x
            self.game_score = SMB.get_mario_score(ram)
            # If we made it further, reset stats
            if self.x_dist > self.farthest_x:
                self.farthest_x = self.x_dist
                self._frames_since_progress = 0
            else:
                self._frames_since_progress += 1

            #@TODO: set this as part of config
            if self._frames_since_progress > 60:
                self.is_alive = False
                return False

            # print(SMB.get_mario_location_in_level(ram).x)
            
        else:
            return False

        if ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2:
            self.is_alive = False
            return False

        self.set_input_as_array(ram, tiles)

        # Calculate the output
        output = self.network.feed_forward(self.inputs_as_array)
        threshold = np.where(output > 0.5)[0]  # @TODO: Maybe make threshold part of config?
        self.buttons_to_press.fill(0)  # Clear

        # Set buttons
        for b in threshold:
            self.buttons_to_press[ouput_to_buttons_map[b]] = 1

        return True
    
def save_mario(population_folder: str, individual_name: str, mario: Mario) -> None:
    # Make population folder if it doesnt exist
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # Save settings.config
    if 'settings.config' not in os.listdir(population_folder):
        with open(os.path.join(population_folder, 'settings.config'), 'w') as config_file:
            config_file.write(mario.config._config_text_file)
    
    # Make a directory for the individual
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    L = len(mario.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = mario.network.params[w_name]
        bias = mario.network.params[b_name]

        np.save(os.path.join(individual_dir, w_name), weights)
        np.save(os.path.join(individual_dir, b_name), bias)

def get_num_inputs(config: Config) -> int:
    u, d, l, r = config.NeuralNetwork.inputs_size
    ud = int(bool(u and d))  # If both u and d directions are non-zero, there is an additional square (Mario)
    lr = int(bool(l and r))  # If both l and r directions are non-zero, there is an additional square (Mario)
    num_inputs = (u + d + ud) * (l + r + lr)
    return num_inputs

def get_num_trainable_parameters(config: Config) -> int:
    num_inputs = get_num_inputs(config)
    hidden_layers = config.NeuralNetwork.hidden_network_architecture
    num_outputs = 6  # U, D, L, R, A, B

    layers = [num_inputs] + hidden_layers + [num_outputs]
    num_params = 0
    for i in range(0, len(layers)-1):
        L      = layers[i]
        L_next = layers[i+1]
        num_params += L*L_next + L_next

    return num_params

