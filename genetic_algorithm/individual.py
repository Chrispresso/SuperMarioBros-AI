 
from abc import abstractmethod
from typing import Optional, Union
import numpy as np

class Individual(object):
    def __init__(self):
        pass

    @abstractmethod
    def calculate_fitness(self):
        raise Exception('calculate_fitness function must be defined')

    @property
    @abstractmethod
    def fitness(self):
        raise Exception('fitness property must be defined')

    @fitness.setter
    @abstractmethod
    def fitness(self, val):
        raise Exception('fitness property cannot be set. Use calculate_fitness instead')

    @abstractmethod
    def encode_chromosome(self):
        raise Exception('encode_chromosome function must be defined')

    @abstractmethod
    def decode_chromosome(self):
        raise Exception('decode_chromosome function must be defined')

    @property
    @abstractmethod
    def chromosome(self):
        raise Exception('chromosome property must be defined')

    @chromosome.setter
    def chromosome(self, val):
        raise Exception('chromosome property cannot be set.')