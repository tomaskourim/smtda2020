# support functions
import logging
from datetime import datetime
from typing import List


def bernoulli2ising(bernoulli: int) -> int:
    """
    Transfers variable form Bernoulli random to Ising random; see
    https://en.wikipedia.org/wiki/Bernoulli_distribution
    https://en.wikipedia.org/wiki/Ising_model

    :param bernoulli: int
    :return:
    """
    if bernoulli == 1:
        return 1
    elif bernoulli == 0:
        return -1
    else:
        raise Exception(f'Unexpected value of Bernoulli distribution: {bernoulli}')


def get_current_probability(c_lambdas: List[float], last_probability: float, step: int, walk_type: str) -> float:
    """
    Computes the transition probability for the next step according to the respective definition as in the paper.
    :param c_lambdas:
    :param last_probability:
    :param step: as Ising variable
    :param walk_type:
    :return:
    """
    if step == '' or step == 0:  # at the beginning of the walk just return p0
        return last_probability
    if walk_type == 'success_punished':
        return (c_lambdas[0]) * last_probability + (0.5 * (1 - c_lambdas[0]) * (1 - step))
    elif walk_type == 'success_rewarded':
        return (c_lambdas[0]) * last_probability + (0.5 * (1 - c_lambdas[0]) * (1 + step))
    elif walk_type == 'success_punished_two_lambdas':
        return (0.5) * (((1 + step) * c_lambdas[0]) * last_probability + (1 - step) * (
                (1) - (c_lambdas[1]) * ((1) - last_probability)))
    elif walk_type == 'success_rewarded_two_lambdas':
        return (0.5) * (((1 - step) * c_lambdas[0]) * last_probability + ((1 + step) * (
                (1) - (c_lambdas[1]) * ((1) - last_probability))))
    else:
        raise Exception(f'Unexpected walk type: {walk_type}')


class CompleteWalk:
    def __init__(self, probabilities, steps, development):
        self.probabilities = probabilities
        self.steps = steps
        self.development = development


def exp_p_t_array(step_count: int, p0: float, c_lambda: float, model_type: str) -> List[float]:
    e_array = []
    for step in range(0, step_count + 1):
        e_array.append(expected_p_t(step, p0, c_lambda, model_type))
    return e_array


def expected_p_t(step: int, p0: float, c_lambda: float, model_type: str) -> float:
    """
    Computes expected value of transition probability according to theoretical results.
    :param step:
    :param p0:
    :param c_lambda:
    :param model_type:
    :return:
    """
    if model_type == 'success_punished':
        e = (2 * c_lambda - 1) ** step * p0 + (1 - (2 * c_lambda - 1) ** step) / 2 if c_lambda != 0.5 else 0.5
    elif model_type == 'success_rewarded':
        e = p0
    elif model_type == 'success_punished_two_lambdas':
        e = 0
    elif model_type == 'success_rewarded_two_lambdas':
        e = 0
    else:
        raise Exception(f'Unexpected walk type: {model_type}')
    return e


# def support_k(i: int, p0: float, c_lambda: float):
#     return (expected_p_t(i, p0, c_lambda, "success_punished") * (-3 * c_lambda ** 2 + 4 * c_lambda - 1) + (
#             1 - c_lambda) ** 2)


# def expected_p_t_squared_support_sum(step: int, p0: float, c_lambda: float, model_type: str) -> float:
#     e = 0
#     for i in range(0, step):
#         if model_type == 'success_punished':
#             summand = support_k(i, p0, c_lambda) * (3 * c_lambda ** 2 - 2 * c_lambda) ** (step - 1 - i)
#         else:
#             raise Exception(f'Unexpected walk type: {model_type}')
#         e = e + summand
#     return e


# def expected_p_t_squared(step: int, p0: float, c_lambda: float, model_type: str) -> float:
#     """
#     Support function to get the variance
#     :param step:
#     :param p0:
#     :param c_lambda:
#     :param model_type:
#     :return:
#     """
#
#     if model_type == 'success_punished':
#         support_sum = expected_p_t_squared_support_sum(step, p0, c_lambda, model_type)
#         e = p0 ** 2 * (3 * c_lambda ** 2 - 2 * c_lambda) ** step + support_sum if c_lambda != 2 / 3 else 0
#     elif model_type == 'success_rewarded':
#         e = p0 * ((2 * c_lambda - c_lambda ** 2) ** step * (p0 - 1) + 1)
#     elif model_type == 'success_punished_two_lambdas':
#         e = 0
#     elif model_type == 'success_rewarded_two_lambdas':
#         e = 0
#     else:
#         raise Exception(f'Unexpected walk type: {model_type}')
#     return e


def expected_p_t_squared_support_sum(step: int, c_lambda: float, model_type: str) -> float:
    support_sum = 0
    for i in range(0, step):
        if model_type == 'success_punished':
            summand = (2 * c_lambda - 1) ** i * (3 * c_lambda ** 2 - 2 * c_lambda) ** (step - 1 - i)
        else:
            raise Exception(f'Unexpected walk type: {model_type}')
        support_sum = support_sum + summand
    return support_sum


def expected_p_t_squared(step: int, p0: float, c_lambda: float, model_type: str) -> float:
    """
    Support function to get the variance
    :param step:
    :param p0:
    :param c_lambda:
    :param model_type:
    :return:
    """

    if model_type == 'success_punished':
        support_sum = expected_p_t_squared_support_sum(step, c_lambda, model_type)
        e = p0 ** 2 * (3 * c_lambda ** 2 - 2 * c_lambda) ** step + (1 - (3 * c_lambda ** 2 - 2 * c_lambda) ** step) / (
                3 * c_lambda + 1) * (c_lambda + 1) / 2 - (p0 - 1 / 2) * (3 * c_lambda - 1) * (
                    c_lambda - 1) * support_sum if c_lambda != -1 / 3 else p0 ** 2
    elif model_type == 'success_rewarded':
        e = p0 * ((2 * c_lambda - c_lambda ** 2) ** step * (p0 - 1) + 1)
    elif model_type == 'success_punished_two_lambdas':
        e = 0
    elif model_type == 'success_rewarded_two_lambdas':
        e = 0
    else:
        raise Exception(f'Unexpected walk type: {model_type}')
    return e


def var_p_t_array(step_count: int, p0: float, c_lambda: float, model_type: str) -> List[float]:
    var_array = [p0 * (1 - p0)]  # VarP(0)
    for step in range(1, step_count + 1):
        ep2 = expected_p_t_squared(step, p0, c_lambda, model_type)
        ep = expected_p_t(step, p0, c_lambda, model_type)
        var_array.append(ep2 - ep ** 2)
    return var_array


def exp_s_t(step: int, p0: float, c_lambda: float, s0: int, model_type: str) -> float:
    """
    Computes expected value of the position of the walker according to theoretical results.
    :param step:
    :param p0:
    :param c_lambda:
    :param s0:
    :param model_type:
    :return:
    """
    if model_type == 'success_punished':
        e = s0 + (2 * p0 - 1) * (1 - (2 * c_lambda - 1) ** step) / (2 * (1 - c_lambda)) if c_lambda != 0.5 else 0
    elif model_type == 'success_rewarded':
        e = s0 + step * (2 * p0 - 1)
    elif model_type == 'success_punished_two_lambdas':
        e = 0
    elif model_type == 'success_rewarded_two_lambdas':
        e = 0
    else:
        raise Exception(f'Unexpected walk type: {model_type}')
    return e


def e_p_s_t_support_q(iteration: int, p0: float, c_lambda: float, s0: int, model_type: str) -> float:
    a = (1 - c_lambda) * exp_s_t(iteration, p0, c_lambda, s0, model_type)
    b = 2 * c_lambda * expected_p_t_squared(iteration, p0, c_lambda, model_type)
    c = (1 - 2 * c_lambda) * expected_p_t(iteration, p0, c_lambda, model_type)
    d = c_lambda - 1
    return a + b + c + d


def e_p_s_t_support_sum(step: int, p0: float, c_lambda: float, s0: int, model_type: str) -> float:
    support_sum = 0
    for i in range(0, step):
        if model_type == 'success_punished':
            summand = (2 * c_lambda - 1) ** i * e_p_s_t_support_q(step - 1 - i, p0, c_lambda, s0, model_type)
        else:
            raise Exception(f'Unexpected walk type: {model_type}')
        support_sum = support_sum + summand
    return support_sum


def exp_p_s_t(step: int, p0: float, c_lambda: float, s0: int, model_type: str) -> float:
    if model_type == 'success_punished':
        sup_sum = e_p_s_t_support_sum(step, p0, c_lambda, s0, model_type)
        e = p0 * s0 * (2 * c_lambda - 1) ** step + sup_sum
    elif model_type == 'success_rewarded':
        a = 2 * c_lambda * p0 * (p0 - 1) / (1 - c_lambda) ** 2
        e = p0 * s0 + a * (1 - (2 * c_lambda - c_lambda ** 2) ** step) + p0 * step
    elif model_type == 'success_punished_two_lambdas':
        e = 0
    elif model_type == 'success_rewarded_two_lambdas':
        e = 0
    else:
        raise Exception(f'Unexpected walk type: {model_type}')
    return e


def e_s2_t_support_sum(step: int, p0: float, c_lambda: float, s0: int, model_type: str):
    support_sum = 0
    for i in range(0, step):
        if model_type == 'success_punished':
            summand = 4 * exp_p_s_t(i, p0, c_lambda, s0, model_type) - ((2 * p0 - 1) * (
                    1 - (2 * c_lambda - 1) ** i) / (1 - c_lambda) if c_lambda != 1 else 0)
        else:
            raise Exception(f'Unexpected walk type: {model_type}')
        support_sum = support_sum + summand
    return support_sum


def exp_s_t_squared(step: int, p0: float, c_lambda: float, s0: int, model_type: str) -> float:
    if model_type == 'success_punished':
        sup_sum = e_s2_t_support_sum(step, p0, c_lambda, s0, model_type)
        e = s0 ** 2 + step * (1 - 2 * s0) + sup_sum
    elif model_type == 'success_rewarded':
        e = 0
    elif model_type == 'success_punished_two_lambdas':
        e = 0
    elif model_type == 'success_rewarded_two_lambdas':
        e = 0
    else:
        raise Exception(f'Unexpected walk type: {model_type}')
    return e


def exp_s_t_array(step_count: int, p0: float, c_lambda: float, s0: int, model_type: str) -> List[float]:
    e_array = []
    for step in range(0, step_count + 1):
        e_array.append(exp_s_t(step, p0, c_lambda, s0, model_type))
    return e_array


def exp_p_s_t_array(step_count: int, p0: float, c_lambda: float, s0: int, model_type: str) -> List[float]:
    e_array = []
    for step in range(0, step_count + 1):
        e_array.append(exp_p_s_t(step, p0, c_lambda, s0, model_type))
    return e_array


def var_s_t_array(step_count: int, p0: float, c_lambda: float, s0: int, model_type: str) -> List[float]:
    var_array = [0]  # VarS(0) = 0
    for step in range(1, step_count + 1):
        es2 = exp_s_t_squared(step, p0, c_lambda, s0, model_type)
        es = exp_s_t(step, p0, c_lambda, s0, model_type)
        var_array.append(es2 - es ** 2)
        print(step)
    return var_array


def create_logger():
    start_time = datetime.now()

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel('DEBUG')

    # Create handlers
    total_handler = logging.FileHandler('logfile_total.log', mode='w')
    info_handler = logging.FileHandler('logfile_info.log')
    error_handler = logging.FileHandler('logfile_error.log')
    stdout_handler = logging.StreamHandler()

    total_handler.setLevel(logging.DEBUG)
    info_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.WARNING)
    stdout_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    logging_format = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(message)s')
    total_handler.setFormatter(logging_format)
    info_handler.setFormatter(logging_format)
    error_handler.setFormatter(logging_format)
    stdout_handler.setFormatter(logging_format)

    # Add handlers to the logger
    logger.addHandler(total_handler)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(stdout_handler)

    return start_time, logger


def log_time(starting_time, logger):
    end_time = datetime.now()
    logger.info(f"Duration: {(end_time - starting_time)}")
