from typing import List

import matplotlib.pyplot as plt
import numpy as np

from python.support import MultiWalk


def vykresli_ept(steps: int, walks: int):
    t = np.arange(0., steps, 1)

    c_lambdas = [0.99, 0.9, 0.5]
    p0s = [0.5, 0.4]

    plt_rows = 1
    plt_colums = 2
    plt_styles = ['g-.', 'r-.', 'b-.']

    for p_index, p0 in enumerate(p0s):
        plt.subplot(plt_rows, plt_colums, p_index + 1)
        plt.axis([0, steps, -0.1, 1.1])
        plt.title(r'$\tilde{p}=%.2f$' % p0)
        plt.xlabel('t')
        plt.ylabel('Ep(t)')
        for index, c_lambda in enumerate(c_lambdas):
            probs = get_walks(steps, walks, c_lambda, p0)
            meanp = np.mean(probs, axis=0)
            plt.plot(meanp, plt_styles[index], label=r'$\lambda=%.2f$' % c_lambda)
            plt.plot(t, expected_p_t(t, p0, c_lambda), color="k", linewidth=0.5)
        plt.legend(loc='best', fontsize='medium')

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.show()
    fig.savefig(f'ept_walks_{walks}_steps{steps}.pdf', dpi=100)


def vykresli_EXt(steps, walks):
    t = np.arange(0., steps, 1)

    # iteruju pres walks
    lambda0 = 0.99
    p1 = 0.8
    walks_generated = {}
    normalni = []
    turban = []
    moje = []
    probs = []
    step_sizes = []
    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        normalni.append(list(walks_generated[i][2].values()))
        turban.append(list(walks_generated[i][3].values()))
        moje.append(list(walks_generated[i][4].values()))
        probs.append(list(walks_generated[i][0].values()))
        step_sizes.append(list(walks_generated[i][1].values()))

    mean_normal = np.mean(normalni, axis=0)
    mean_turban = np.mean(turban, axis=0)
    mean_moje = np.mean(moje, axis=0)

    #    plt.plot(mean_normal, 'r--', label="mean_normal")
    plt.subplot(321)
    plt.plot(mean_turban, 'b:', label="Variable step size")
    plt.plot(mean_moje, 'g-.', label="Variable probability")
    plt.plot(t, expected_s_t(t, p1, lambda0), 'r-', label="Variable probability - computed")
    #  plt.xlabel('t')
    plt.ylabel('EX(t)')
    plt.legend(loc='best', fontsize='medium')
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)

    lambda0 = 0.9
    p1 = 0.8
    walks_generated = {}
    normalni = []
    turban = []
    moje = []
    probs = []
    step_sizes = []
    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        normalni.append(list(walks_generated[i][2].values()))
        turban.append(list(walks_generated[i][3].values()))
        moje.append(list(walks_generated[i][4].values()))
        probs.append(list(walks_generated[i][0].values()))
        step_sizes.append(list(walks_generated[i][1].values()))

    mean_normal = np.mean(normalni, axis=0)
    mean_turban = np.mean(turban, axis=0)
    mean_moje = np.mean(moje, axis=0)

    #    plt.plot(mean_normal, 'r--', label="mean_normal")
    plt.subplot(323)
    plt.plot(mean_turban, 'b:', label="Variable step size")
    plt.plot(mean_moje, 'g-.', label="Variable probability")
    plt.plot(t, expected_s_t(t, p1, lambda0), 'r-', label="Variable probability - computed")
    # plt.xlabel('t')
    plt.ylabel('EX(t)')
    plt.legend(loc='best', fontsize='medium')
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)

    lambda0 = 0.2
    p1 = 0.8
    walks_generated = {}
    normalni = []
    turban = []
    moje = []
    probs = []
    step_sizes = []
    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        normalni.append(list(walks_generated[i][2].values()))
        turban.append(list(walks_generated[i][3].values()))
        moje.append(list(walks_generated[i][4].values()))
        probs.append(list(walks_generated[i][0].values()))
        step_sizes.append(list(walks_generated[i][1].values()))

    mean_normal = np.mean(normalni, axis=0)
    mean_turban = np.mean(turban, axis=0)
    mean_moje = np.mean(moje, axis=0)

    #    plt.plot(mean_normal, 'r--', label="mean_normal")
    plt.subplot(325)
    plt.plot(mean_turban, 'b:', label="Variable step size")
    plt.plot(mean_moje, 'g-.', label="Variable probability")
    plt.plot(t, expected_s_t(t, p1, lambda0), 'r-', label="Variable probability - computed")
    plt.xlabel('t')
    plt.ylabel('EX(t)')
    plt.legend(loc='best', fontsize='medium')
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)

    # iteruju pres walks
    lambda0 = 0.99
    p1 = 0.5
    walks_generated = {}
    normalni = []
    turban = []
    moje = []
    probs = []
    step_sizes = []
    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        normalni.append(list(walks_generated[i][2].values()))
        turban.append(list(walks_generated[i][3].values()))
        moje.append(list(walks_generated[i][4].values()))
        probs.append(list(walks_generated[i][0].values()))
        step_sizes.append(list(walks_generated[i][1].values()))

    mean_normal = np.mean(normalni, axis=0)
    mean_turban = np.mean(turban, axis=0)
    mean_moje = np.mean(moje, axis=0)

    #    plt.plot(mean_normal, 'r--', label="mean_normal")
    plt.subplot(322)
    plt.plot(mean_turban, 'b:', label="Variable step size")
    plt.plot(mean_moje, 'g-.', label="Variable probability")
    plt.plot(t, expected_s_t(t, p1, lambda0), 'r-', label="Variable probability - computed")
    #  plt.xlabel('t')
    plt.ylabel('EX(t)')
    plt.legend(loc='best', fontsize='medium')
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)

    lambda0 = 0.9
    walks_generated = {}
    normalni = []
    turban = []
    moje = []
    probs = []
    step_sizes = []
    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        normalni.append(list(walks_generated[i][2].values()))
        turban.append(list(walks_generated[i][3].values()))
        moje.append(list(walks_generated[i][4].values()))
        probs.append(list(walks_generated[i][0].values()))
        step_sizes.append(list(walks_generated[i][1].values()))

    mean_normal = np.mean(normalni, axis=0)
    mean_turban = np.mean(turban, axis=0)
    mean_moje = np.mean(moje, axis=0)

    #    plt.plot(mean_normal, 'r--', label="mean_normal")
    plt.subplot(324)
    plt.plot(mean_turban, 'b:', label="Variable step size")
    plt.plot(mean_moje, 'g-.', label="Variable probability")
    plt.plot(t, expected_s_t(t, p1, lambda0), 'r-', label="Variable probability - computed")
    # plt.xlabel('t')
    plt.ylabel('EX(t)')
    plt.legend(loc='best', fontsize='medium')
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)

    lambda0 = 0.2
    walks_generated = {}
    normalni = []
    turban = []
    moje = []
    probs = []
    step_sizes = []
    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        normalni.append(list(walks_generated[i][2].values()))
        turban.append(list(walks_generated[i][3].values()))
        moje.append(list(walks_generated[i][4].values()))
        probs.append(list(walks_generated[i][0].values()))
        step_sizes.append(list(walks_generated[i][1].values()))

    mean_normal = np.mean(normalni, axis=0)
    mean_turban = np.mean(turban, axis=0)
    mean_moje = np.mean(moje, axis=0)

    #    plt.plot(mean_normal, 'r--', label="mean_normal")
    plt.subplot(326)
    plt.plot(mean_turban, 'b:', label="Variable step size")
    plt.plot(mean_moje, 'g-.', label="Variable probability")
    plt.plot(t, expected_s_t(t, p1, lambda0), 'r-', label="Variable probability - computed")
    plt.xlabel('t')
    plt.ylabel('EX(t)')
    plt.legend(loc='best', fontsize='medium')
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)

    plt.show()

    plt.savefig('examples_p0=%.2f.pdf' % p1, bbox_inches='tight')


def vykresli_prubehy():
    walks_generated = {}

    i = 0
    lambda0 = 0.99
    p1 = 0.45
    steps = 1000
    walks_generated[i] = generate_walk(steps, p1, lambda0)

    x, y = zip(*sorted(walks_generated[i][2].items()))
    a, b = zip(*sorted(walks_generated[i][3].items()))
    c, d = zip(*sorted(walks_generated[i][4].items()))
    plt.subplot(221)
    plt.plot(x, y, 'r--', label="Standard RW")
    plt.plot(a, b, 'b:', label="Variable step size")
    plt.plot(c, d, 'g-.', label="Variable probability")
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.legend(loc='best', fontsize='medium')

    i = 1
    lambda0 = 0.8
    walks_generated[i] = generate_walk(steps, p1, lambda0)

    x, y = zip(*sorted(walks_generated[i][2].items()))
    a, b = zip(*sorted(walks_generated[i][3].items()))
    c, d = zip(*sorted(walks_generated[i][4].items()))
    plt.subplot(222)
    plt.plot(x, y, 'r--', label="Standard RW")
    plt.plot(a, b, 'b:', label="Variable step size")
    plt.plot(c, d, 'g-.', label="Variable probability")
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.legend(loc='best', fontsize='medium')

    i = 2
    lambda0 = 0.5
    walks_generated[i] = generate_walk(steps, p1, lambda0)

    x, y = zip(*sorted(walks_generated[i][2].items()))
    a, b = zip(*sorted(walks_generated[i][3].items()))
    c, d = zip(*sorted(walks_generated[i][4].items()))
    plt.subplot(223)
    plt.plot(x, y, 'r--', label="Standard RW")
    plt.plot(a, b, 'b:', label="Variable step size")
    plt.plot(c, d, 'g-.', label="Variable probability")
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.legend(loc='best', fontsize='medium')

    i = 3
    lambda0 = 0.2
    walks_generated[i] = generate_walk(steps, p1, lambda0)

    x, y = zip(*sorted(walks_generated[i][2].items()))
    a, b = zip(*sorted(walks_generated[i][3].items()))
    c, d = zip(*sorted(walks_generated[i][4].items()))
    plt.subplot(224)
    plt.plot(x, y, 'r--', label="Standard RW")
    plt.plot(a, b, 'b:', label="Variable step size")
    plt.plot(c, d, 'g-.', label="Variable probability")
    plt.title(r'$\lambda=%.2f$' % lambda0 + ", " + r'$\tilde{p}=%.2f$' % p1)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.legend(loc='best', fontsize='medium')

    mng = plt.get_current_fig_manager()
    ### works on Ubuntu??? >> did NOT working on windows
    # mng.resize(*mng.window.maxsize())
    mng.window.state('zoomed')

    plt.savefig('examples_p0=%.2f.pdf' % p1, bbox_inches='tight')


def get_walks(steps: int, walks: int, c_lambda: float, p0: float) -> List[list]:
    probs = []
    for i in range(0, walks):
        probs.append(list(generate_walk(steps, p0, c_lambda).probs.values()))
    return probs


def vykresli_varpt(steps: int, walks: int):
    c_lambdas = [0.99, 0.9, 0.5]
    p0s = [0.8, 0.5, 0.1]

    plt_rows = 1
    plt_colums = 3
    plt_styles = ['g-.', 'r-.', 'b-.']

    for p_index, p0 in enumerate(p0s):
        plt.subplot(plt_rows, plt_colums, p_index + 1)
        plt.axis([0, steps, 0, 0.07])
        plt.title(r'$\tilde{p}=%.2f$' % p0)
        plt.xlabel('t')
        plt.ylabel('Var p(t)')
        for index, c_lambda in enumerate(c_lambdas):
            probs = get_walks(steps, walks, c_lambda, p0)
            varp = np.var(probs, axis=0)
            plt.plot(varp, plt_styles[index], label=r'$\lambda=%.2f$' % c_lambda)
        plt.legend(loc='best', fontsize='medium')

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.show()
    fig.savefig(f'varpt_walks_{walks}_steps{steps}.pdf', dpi=100)


def generate_walk(steps, p0, c_lambda) -> MultiWalk:
    ## tohle vracim
    probs = {}
    step_sizes = {}
    normal_prubehy = {}
    turban_prubehy = {}
    ja_prubehy = {}

    ## pomocne hodnoty
    ja_kroky = {}
    turban_kroky = {}
    normal_kroky = {}

    ## nulte hodnoty
    probs[0] = p0
    step_sizes[0] = 1

    ## nahodna pro vsechny stejna pravedpodobnost
    random_number_0_1 = np.random.uniform()

    ja_kroky[0] = 1 if random_number_0_1 < p0 else -1
    turban_kroky[0] = 1 if random_number_0_1 < p0 else -1
    normal_kroky[0] = 1 if random_number_0_1 < p0 else -1

    ja_prubehy[0] = ja_kroky[0]
    turban_prubehy[0] = turban_kroky[0]
    normal_prubehy[0] = normal_kroky[0]
    # print("Prav gen: ", random_number_0_1, "Prav hranice:", p0, "Ja krok: ", ja_kroky[0], "Turban krok: ", turban_kroky[0],
    #       "Normal krok: ", normal_kroky[0])
    # iteruju pres steps
    for i in range(1, steps):
        probs[i] = c_lambda * probs[i - 1] + (1 / 2) * (1 - c_lambda) * (1 - ja_kroky[i - 1])
        step_sizes[i] = c_lambda * step_sizes[i - 1] + (1 - c_lambda) * (1 - normal_kroky[i - 1])

        ## nahodna pro vsechny stejna pravedpodobnost
        random_number_0_1 = np.random.uniform()

        ja_kroky[i] = 1 if random_number_0_1 < probs[i] else -1
        normal_kroky[i] = 1 if random_number_0_1 < p0 else -1
        turban_kroky[i] = step_sizes[i] + normal_kroky[i] - 1

        ja_prubehy[i] = ja_prubehy[i - 1] + ja_kroky[i]
        turban_prubehy[i] = turban_prubehy[i - 1] + turban_kroky[i]
        normal_prubehy[i] = normal_prubehy[i - 1] + normal_kroky[i]
        # print("Prav gen: ", random_number_0_1, "Prav hranice:", probs[i], "Ja krok: ", ja_kroky[i], "Turban krok: ",
        #       turban_kroky[i], "Normal krok: ", normal_kroky[i])

    return MultiWalk(probs, step_sizes, normal_prubehy, turban_prubehy, ja_prubehy, steps)


def expected_s_t(t, p0, c_lambda):
    return (2 * p0 - 1) * (1 - np.power(2 * c_lambda - 1, t)) / (
            2 * (1 - c_lambda)) if c_lambda != 0.5 else [0.5] * len(t)


def expected_p_t(t, p0, c_lambda):
    e = np.power(2 * c_lambda - 1, t - 1) * p0 + (
            1 - np.power(2 * c_lambda - 1, t - 1)) / 2 if c_lambda != 0.5 else [0.5] * len(t)
    return e


def generate_walks(steps, walks, p1, lambda0):
    # iteruju pres walks
    lambda0 = 0.99
    p1 = 0.8
    walks_generated = {}
    moje = []

    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        moje.append(list(walks_generated[i][4].values()))

    mean_moje = np.var(moje, axis=0)

    plt.subplot(121)

    plt.plot(mean_moje, 'g-.', label=r'$\lambda=%.2f$' % lambda0)

    lambda0 = 0.9
    walks_generated = {}
    moje = []

    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        moje.append(list(walks_generated[i][4].values()))

    mean_moje = np.var(moje, axis=0)

    plt.plot(mean_moje, 'r-.', label=r'$\lambda=%.2f$' % lambda0)

    lambda0 = 0.5
    walks_generated = {}
    moje = []

    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        moje.append(list(walks_generated[i][4].values()))

    mean_moje = np.var(moje, axis=0)

    plt.plot(mean_moje, 'b-.', label=r'$\lambda=%.2f$' % lambda0)

    plt.xlabel('t')
    plt.ylabel('Var X(t)')
    plt.legend(loc='best', fontsize='medium')
    plt.axis([0, steps, 0, 1000])
    plt.title(r'$\tilde{p}=%.2f$' % p1)

    # dalsi p0
    lambda0 = 0.99
    p1 = 0.5
    walks_generated = {}
    moje = []

    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        moje.append(list(walks_generated[i][4].values()))

    mean_moje = np.var(moje, axis=0)

    plt.subplot(122)

    plt.plot(mean_moje, 'g-.', label=r'$\lambda=%.2f$' % lambda0)

    lambda0 = 0.9
    walks_generated = {}
    moje = []

    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        moje.append(list(walks_generated[i][4].values()))

    mean_moje = np.var(moje, axis=0)

    plt.plot(mean_moje, 'r-.', label=r'$\lambda=%.2f$' % lambda0)

    lambda0 = 0.5
    walks_generated = {}
    moje = []

    for i in range(0, walks):
        walks_generated[i] = generate_walk(steps, p1, lambda0)
        moje.append(list(walks_generated[i][4].values()))

    mean_moje = np.var(moje, axis=0)

    plt.plot(mean_moje, 'b-.', label=r'$\lambda=%.2f$' % lambda0)

    plt.xlabel('t')
    plt.ylabel('Var X(t)')
    plt.legend(loc='best', fontsize='medium')
    plt.axis([0, steps, 0, 1000])
    plt.title(r'$\tilde{p}=%.2f$' % p1)

    plt.show()

    plt.savefig('VarXt.pdf', bbox_inches='tight')

    pass


if __name__ == "__main__":
    # generate_walks(1000, 500, 0.6, 0.95)
    vykresli_ept(100, 100)
    # vykresli_EXt(1000, 100)
    vykresli_varpt(100, 100)
    print("hotovo")
