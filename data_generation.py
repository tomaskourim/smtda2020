# code used to generate data using theoretical models.

import os
import pickle
from typing import List

import numpy as np

from common import bernoulli2ising, get_current_probability, CompleteWalk, create_logger, log_time
from config import C_LAMBDAS, START_PROBABILITIES, STEP_COUNTS, C_LAMBDA_PAIRS, DATA_DIRNAME, MODEL_TYPES, \
    REPETITIONS_OF_WALK_S, REPETITIONS_OF_WALK_SERIES


def generate_random_walk(model_type: str, starting_probability: float, c_lambdas: List[float], walk_steps: int) -> \
        CompleteWalk:
    steps = [0]  # in the model, probabilities start with p0, but steps with x1
    development = [0]
    probabilities = [starting_probability]
    for i in range(1, walk_steps + 1):
        # next step using actual probability
        steps.append(bernoulli2ising(np.random.binomial(1, probabilities[i - 1], 1)[0]))
        development.append(development[i - 1] + steps[i])
        probabilities.append(get_current_probability(c_lambdas, probabilities[i - 1], steps[i], model_type))
    return CompleteWalk(probabilities, steps, development)


def generate_random_walks(model_type: str, starting_probability: float, c_lambdas: List[float], walk_steps: int,
                          repetitions: int) -> List[CompleteWalk]:
    complete_walks = []
    for j in range(0, repetitions):
        complete_walks.append(generate_random_walk(model_type, starting_probability, c_lambdas, walk_steps))
    return complete_walks


def save_walks(walks: List[List[int]], model_type: str, starting_probability: float, c_lambdas: List[float],
               step_count: int, repetitions_of_walk: int, repetition: int):
    if not os.path.exists(DATA_DIRNAME):
        os.mkdir(DATA_DIRNAME)
    filename = f"{DATA_DIRNAME}/K{repetitions_of_walk}/{model_type}__start{starting_probability}__lambdas{c_lambdas}__steps{step_count}__repetition{repetition}.pkl"
    with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([walks, model_type, starting_probability, c_lambdas, step_count, repetition], f)


def list_walks2list_lists(walks):
    walks_steps = []
    walks_developments = []
    walks_probabilities = []
    for walk in walks:
        walks_steps.append(walk.steps)
        walks_developments.append(walk.development)
        walks_probabilities.append(walk.probabilities)
    return [walks_probabilities, walks_steps, walks_developments]


def generate_and_save_walks(model_type: str, starting_probability: float, c_lambdas: List[float], step_count: int,
                            repetitions_of_walk: int, repetition: int) -> List[List[int]]:
    walks = generate_random_walks(model_type, starting_probability, c_lambdas, step_count, repetitions_of_walk)
    walks_steps = list_walks2list_lists(walks)[1]
    save_walks(walks_steps, model_type, starting_probability, c_lambdas, step_count, repetitions_of_walk, repetition)
    return walks_steps


def main():
    # try different Random Walk with Varying Transition Probabilities definitions
    # different lambdas, starting probability, number of steps, multiple times with same starting variables
    # iterate over lambdas, starting probability, number of steps (reward might lead to floating point errors),
    # repetitions
    # save into .csv?

    for repetition in range(0, REPETITIONS_OF_WALK_SERIES):
        for index, c_lambda in enumerate(C_LAMBDAS):
            for starting_probability in START_PROBABILITIES:
                for step_count in STEP_COUNTS:
                    for model_type in MODEL_TYPES:
                        if 'two_lambdas' in model_type:
                            c_lambdas = C_LAMBDA_PAIRS[index]
                        else:
                            c_lambdas = [c_lambda]
                        generate_and_save_walks(model_type, starting_probability, c_lambdas, step_count,
                                                REPETITIONS_OF_WALK_S[0], repetition)


if __name__ == '__main__':
    start_time, logger = create_logger()
    main()
    log_time(start_time, logger)
