import numpy as np
from typing import List, Union, Optional
from .individual import Individual


def gaussian_mutation(chromosome: np.ndarray, prob_mutation: float, 
                      mu: List[float] = None, sigma: List[float] = None,
                      scale: Optional[float] = None) -> None:
    """
    Perform a gaussian mutation for each gene in an individual with probability, prob_mutation.
    If mu and sigma are defined then the gaussian distribution will be drawn from that,
    otherwise it will be drawn from N(0, 1) for the shape of the individual.
    """
    # Determine which genes will be mutated
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    # If mu and sigma are defined, create gaussian distribution around each one
    if mu and sigma:
        gaussian_mutation = np.random.normal(mu, sigma)
    # Otherwise center around N(0,1)
    else:
        gaussian_mutation = np.random.normal(size=chromosome.shape)
    
    if scale:
        gaussian_mutation[mutation_array] *= scale

    # Update
    chromosome[mutation_array] += gaussian_mutation[mutation_array]

def random_uniform_mutation(chromosome: np.ndarray, prob_mutation: float,
                            low: Union[List[float], float],
                            high: Union[List[float], float]
                            ) -> None:
    """
    Randomly mutate each gene in an individual with probability, prob_mutation.
    If a gene is selected for mutation it will be assigned a value with uniform probability
    between [low, high).
    @Note [low, high) is defined for each gene to help get the full range of possible values
    """
    assert type(low) == type(high), 'low and high must have the same type'
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    if isinstance(low, list):
        uniform_mutation = np.random.uniform(low, high)
    else:
        uniform_mutation = np.random.uniform(low, high, size=chromosome.shape)
    chromosome[mutation_array] = uniform_mutation[mutation_array]

def uniform_mutation_with_respect_to_best_individual(chromosome: np.ndarray, best_chromosome: np.ndarray, prob_mutation: float) -> None:
    """
    Ranomly mutate each gene in an individual with probability, prob_mutation.
    If a gene is selected for mutation it will nudged towards the gene from the best individual.
    """
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    uniform_mutation = np.random.uniform(size=chromosome.shape)
    chromosome[mutation_array] += uniform_mutation[mutation_array] * (best_chromosome[mutation_array] - chromosome[mutation_array])

def cauchy_mutation(individual: np.ndarray, scale: float) -> np.ndarray:
    pass

def exponential_mutation(chromosome: np.ndarray, xi: Union[float, np.ndarray], prob_mutation: float) -> None:
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    # Fill xi if necessary
    if not isinstance(xi, np.ndarray):
        xi_val = xi
        xi = np.empty(chromosome.shape)
        xi.fill(xi_val)

    # Change xi so we get E(0, 1), instead of E(0, xi)
    xi_div = 1.0 / xi
    xi.fill(1.0)
    
    # Eq 11.17
    y = np.random.uniform(size=chromosome.shape)
    x = np.empty(chromosome.shape)
    x[y <= 0.5] = (1.0 / xi[y <= 0.5]) * np.log(2 * y[y <= 0.5])
    x[y > 0.5] = -(1.0 / xi[y > 0.5]) * np.log(2 * (1 - y[y > 0.5]))

    # Eq 11.16
    delta = np.empty(chromosome.shape)
    delta[mutation_array] = (xi[mutation_array] / 2.0) * np.exp(-xi[mutation_array] * np.abs(x[mutation_array]))

    # Update delta such that E(0, xi) = (1 / xi) * E(0 , 1)
    delta[mutation_array] = xi_div[mutation_array] * delta[mutation_array]

    # Update individual
    chromosome[mutation_array] += delta[mutation_array]

def mmo_mutation(chromosome: np.ndarray, prob_mutation: float) -> None:
    from scipy import stats
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    normal = np.random.normal(size=chromosome.shape)  # Eq 11.21
    cauchy = stats.cauchy.rvs(size=chromosome.shape)  # Eq 11.22
    
    # Eq 11.20
    delta = np.empty(chromosome.shape)
    delta[mutation_array] = normal[mutation_array] + cauchy[mutation_array]

    # Update individual
    chromosome[mutation_array] += delta[mutation_array]