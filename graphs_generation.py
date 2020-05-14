# used to generate useful graphics
import matplotlib.pyplot as plt
import numpy as np

from common import exp_p_t_array, var_p_t_array, log_time, create_logger, exp_s_t_array, exp_p_s_t_array, var_s_t_array, \
    exp_x_t_array, var_x_t_array
from config import MODEL_TYPES, REPETITIONS_OF_WALK_S, \
    C_LAMBDAS_TESTING, START_PROBABILITIES_TESTING, STEP_COUNTS_TESTING, C_LAMBDA_PAIRS_TESTING
from data_generation import generate_random_walks, list_walks2list_lists


def main(simulated_property="probability"):
    plt_rows = 1
    plt_columns = len(START_PROBABILITIES_TESTING)
    mean_styles = ['g.', 'r.', 'b.']
    var_styles = ['g-.', 'r-.', 'b-.']
    expected_styles = ['g-', 'r-', 'b-']
    model_min_y = [-3, -10, 0, 0]
    model_max_y = [20, 30, 50, 2000]
    for step_count in STEP_COUNTS_TESTING:
        for repetitions in REPETITIONS_OF_WALK_S:
            for model_index, model_type in enumerate(MODEL_TYPES):
                if 'two_lambdas' in model_type:
                    two_lambda = True
                else:
                    two_lambda = False
                # TODO handle with dignity
                min_y = 0 if simulated_property == "probability" else -1.05 if simulated_property == "step" else \
                    model_min_y[model_index]
                max_y = 1 if simulated_property == "probability" else 1.05 if simulated_property == "step" else \
                    model_max_y[model_index]
                for p_index, starting_probability in enumerate(START_PROBABILITIES_TESTING):
                    plt.subplot(plt_rows, plt_columns, p_index + 1)
                    plt.title(r'$p_{0}=%.2f$' % starting_probability, fontsize=20)
                    plt.axis([1, step_count, min_y, max_y])
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
                                                      repetitions)
                        probabilities, steps, developments = list_walks2list_lists(walks)

                        if simulated_property == "probability":
                            mean = np.mean(probabilities, axis=0)
                            variance = np.var(probabilities, axis=0)
                        elif simulated_property == "position":
                            mean = np.mean(developments, axis=0)
                            variance = np.var(developments, axis=0)
                        elif simulated_property == "p_s":
                            ps_all = []
                            for observation in range(0, repetitions):
                                ps_single = []
                                for walk_step in range(0, step_count + 1):
                                    ps_single.append(
                                        probabilities[observation][walk_step] * developments[observation][walk_step])
                                ps_all.append(ps_single)
                            mean = np.mean(ps_all, axis=0)
                            variance = np.var(ps_all, axis=0)
                        elif simulated_property == "step":
                            mean = np.mean(steps, axis=0)
                            variance = np.var(steps, axis=0)
                        else:
                            raise Exception("unexpected property type")
                        plt.plot(mean, mean_styles[index], label=label)
                        plt.plot(variance, var_styles[index])
                        if not two_lambda:
                            if simulated_property == "probability":
                                plt.plot(exp_p_t_array(step_count, starting_probability, c_lambda, model_type),
                                         expected_styles[index], linewidth=0.7)
                                plt.plot(var_p_t_array(step_count, starting_probability, c_lambda, model_type),
                                         expected_styles[index], linewidth=0.7)
                            elif simulated_property == "position":
                                s0 = 0
                                plt.plot(exp_s_t_array(step_count, starting_probability, c_lambda, s0, model_type),
                                         expected_styles[index], linewidth=0.7)
                                plt.plot(var_s_t_array(step_count, starting_probability, c_lambda, s0, model_type),
                                         expected_styles[index], linewidth=0.7)
                            elif simulated_property == "p_s":
                                s0 = 0
                                plt.plot(exp_p_s_t_array(step_count, starting_probability, c_lambda, s0, model_type),
                                         expected_styles[index], linewidth=0.7)
                            elif simulated_property == "step":
                                plt.plot(exp_x_t_array(step_count, starting_probability, c_lambda, model_type),
                                         expected_styles[index], linewidth=0.7)
                                plt.plot(var_x_t_array(step_count, starting_probability, c_lambda, model_type),
                                         expected_styles[index], linewidth=0.7)
                        plt.legend(loc='best', fontsize='xx-large', markerscale=3)
                        logger.info(
                            f"Type: {simulated_property}, model {model_type}, steps {step_count}, reps {repetitions}, p0 {starting_probability}, lambda {c_lambda}")

                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)
                fig.show()
                fig.savefig(
                    f'e_{simulated_property}_{repetitions}_walks_{step_count}_steps_type_{model_type}.pdf',
                    dpi=100)


if __name__ == '__main__':
    start_time, logger = create_logger()
    # main(simulated_property="position")
    # main(simulated_property="probability")
    main(simulated_property="step")
    # main(simulated_property="p_s")
    log_time(start_time, logger)
