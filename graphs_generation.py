# used to generate useful graphics
import matplotlib.pyplot as plt
import numpy as np

from common import expected_p_t_array, var_p_t_array, log_time, create_logger
from config import MODEL_TYPES, REPETITIONS_OF_WALK_S, \
    C_LAMBDAS_TESTING, START_PROBABILITIES_TESTING, STEP_COUNTS_TESTING, C_LAMBDA_PAIRS_TESTING
from data_generation import generate_random_walks, list_walks2list_lists


def main(simulated_property="probability"):
    plt_rows = 1
    plt_columns = 3
    mean_styles = ['g.', 'r.', 'b.']
    var_styles = ['g-.', 'r-.', 'b-.']
    expected_styles = ['g-', 'r-', 'b-']
    for step_count in STEP_COUNTS_TESTING:
        for model_type in MODEL_TYPES:
            if 'two_lambdas' in model_type:
                two_lambda = True
            else:
                two_lambda = False
            # TODO handle with dignity
            min_y = 0 if simulated_property == "probability" else -3
            max_y = 1 if simulated_property == "probability" else 30
            for p_index, starting_probability in enumerate(START_PROBABILITIES_TESTING):
                plt.subplot(plt_rows, plt_columns, p_index + 1)
                plt.title(r'$p_{0}=%.2f$' % starting_probability, fontsize=20)
                plt.axis([0, step_count, min_y, max_y])
                plt.xlabel('steps', fontsize=18)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                for index, c_lambda in enumerate(C_LAMBDAS_TESTING):
                    if two_lambda:
                        c_lambdas = C_LAMBDA_PAIRS_TESTING[index]
                        label = r'$\bar{\lambda}=[%.2f,%.2f]$' % (c_lambdas[0], c_lambdas[1])
                    else:
                        c_lambdas = [c_lambda]
                        label = r'$\lambda=%.2f$' % c_lambda
                    walks = generate_random_walks(model_type, starting_probability, c_lambdas, step_count,
                                                  REPETITIONS_OF_WALK_S[0])
                    probabilities, steps, developments = list_walks2list_lists(walks)

                    if simulated_property == "probability":
                        mean = np.mean(probabilities, axis=0)
                        variance = np.var(probabilities, axis=0)
                    elif simulated_property == "position":
                        mean = np.mean(developments, axis=0)
                        variance = np.var(developments, axis=0)
                    else:
                        raise Exception("unexpected property type")
                    plt.plot(mean, mean_styles[index], label=label)
                    plt.plot(variance, var_styles[index])
                    if not two_lambda and simulated_property == "probability":
                        plt.plot(expected_p_t_array(step_count, float(starting_probability), c_lambda, model_type),
                                 expected_styles[index], linewidth=0.7)
                        plt.plot(var_p_t_array(step_count, float(starting_probability), c_lambda, model_type),
                                 expected_styles[index], linewidth=0.7)
                    plt.legend(loc='best', fontsize='xx-large', markerscale=3)

            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            fig.show()
            fig.savefig(
                f'e_{simulated_property}_{REPETITIONS_OF_WALK_S[0]}_walks_{step_count}_steps_type_{model_type}.pdf',
                dpi=100)


if __name__ == '__main__':
    start_time, logger = create_logger()
    # main(simulated_property="position")
    main(simulated_property="probability")
    log_time(start_time, logger)
