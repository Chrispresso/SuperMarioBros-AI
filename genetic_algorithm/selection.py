import numpy as np
import random
from typing import List
from .population import Population
from .individual import Individual


def elitism_selection(population: Population, num_individuals: int) -> List[Individual]:
    # Is this efficient? No. What would be better? Max heap. Will I change it? Probably not this time.
    individuals = sorted(population.individuals, key = lambda individual: individual.fitness, reverse=True)
    return individuals[:num_individuals]

def roulette_wheel_selection(population: Population, num_individuals: int) -> List[Individual]:
    selection = []
    wheel = sum(individual.fitness for individual in population.individuals)
    for _ in range(num_individuals):
        pick = random.uniform(0, wheel)
        current = 0
        for individual in population.individuals:
            current += individual.fitness
            if current > pick:
                selection.append(individual)
                break

    return selection

def tournament_selection(population: Population, num_individuals: int, tournament_size: int) -> List[Individual]:
    selection = []
    for _ in range(num_individuals):
        tournament = np.random.choice(population.individuals, tournament_size)
        best_from_tournament = max(tournament, key = lambda individual: individual.fitness)
        selection.append(best_from_tournament)

    return selection