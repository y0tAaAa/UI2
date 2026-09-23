import math
import numpy as np
import random

from numpy.typing import NDArray

def uniform_space(amount_of_genes: int,
                  lower_limit:     float,
                  upper_limit:     float) -> NDArray:
    return np.stack([np.array([lower_limit,]*amount_of_genes),
                     np.array([upper_limit,]*amount_of_genes)])


def selbest(old_pop, old_pop_fit, n_list, reverse=False):

    new_pop, new_fit = [], []

    fit_index = sorted(range(len(old_pop_fit)),
                       key=lambda k: old_pop_fit[k], reverse=reverse)

    for i in range(len(n_list)):
        for j in range(n_list[i]):
            new_pop.append(old_pop[fit_index[i]])
            new_fit.append(old_pop_fit[fit_index[i]])

    return np.array(new_pop), np.array(new_fit)


def selsort(old_pop, old_pop_fit, n, reverse=False):

    new_pop, new_fit = [], []

    fit_index = sorted(range(len(old_pop_fit)),
                       key=lambda k: old_pop_fit[k], reverse=reverse)

    for i in range(n):
        new_pop.append(old_pop[fit_index[i]])
        new_fit.append(old_pop_fit[fit_index[i]])

    return np.array(new_pop), np.array(new_fit)


def seldiv(old_pop, old_pop_fit, n_list, mode, reverse=False):

    shape = old_pop.shape
    new_pop, new_fit = [], []

    if mode == 1:
        reference_list = old_pop[sorted(
            range(len(old_pop_fit)), key=lambda k: old_pop_fit[k], reverse=reverse)[0]]
    elif mode == 0:
        reference_list = np.mean(old_pop, axis=0)

    diversity_list = np.zeros([shape[0]])
    div_index = 0
    for i in range(shape[0]):
        diversity_list[i] = np.sum(np.abs(reference_list - old_pop[i]))

    for i in range(len(n_list)):
        maxi, div_index = 0, 0

        for j in range(shape[0]):
            if (diversity_list[j] > maxi) and (old_pop_fit[j] not in new_fit):
                maxi = diversity_list[j]
                div_index = j

        diversity_list[div_index] = 0

        for j in range(n_list[i]):
            new_pop.append(old_pop[div_index])
            new_fit.append(old_pop_fit[div_index])

    return np.array(new_pop), np.array(new_fit)


def selrand(old_pop, old_pop_fit, n):

    shape = old_pop.shape
    new_pop, new_fit = [], []

    for i in range(n):
        j = math.ceil(random.random() * shape[0]) - 1
        new_pop.append(old_pop[j])
        new_fit.append(old_pop_fit[j])

    return np.array(new_pop), np.array(new_fit)


def selsus(old_pop, old_pop_fit, n, reverse=False):

    shape = old_pop.shape
    new_pop, new_fit = [], []

    if reverse:
        weights_help = np.append(
            1 / ((abs(np.subtract(old_pop_fit, max(old_pop_fit))) + 1) * sum(old_pop_fit)), 0)
    else:
        weights_help = np.append(
            1 / ((np.subtract(old_pop_fit, min(old_pop_fit)) + 1) * sum(old_pop_fit)), 0)

    weights = np.zeros(shape[0] + 1)
    for i in range(shape[0] - 1, -2, -1):
        weights[i] = weights[i + 1] + weights_help[i]
    weights[-1] = 0

    max_weight = max(weights)
    if max_weight == 0:
        max_weight = 0.00001

    weights /= (max_weight / 100)

    roulette_help = random.uniform(0, 1) * (100 / n) - 0.00001
    roulette = np.array([(i * (100 / n) + roulette_help) for i in range(n)])

    for i in range(n):
        for j in range(shape[0]):
            if (roulette[i] <= weights[j]) and (roulette[i] >= weights[j + 1]):
                break

        new_pop.append(old_pop[j])
        new_fit.append(old_pop_fit[j])

    return np.array(new_pop), np.array(new_fit)


def seltourn(old_pop, old_pop_fit, n, reverse=False):

    shape = old_pop.shape
    new_pop, new_pop_fit = [], []

    for i in range(n):

        j = math.ceil(shape[0] * random.random()) - 1
        k = math.ceil(shape[0] * random.random()) - 1

        if j == k:
            new_pop.append(old_pop[j])
            new_pop_fit.append(old_pop_fit[j])

        elif old_pop_fit[j] <= old_pop_fit[k] and not reverse:
            new_pop.append(old_pop[j])
            new_pop_fit.append(old_pop_fit[j])

        elif old_pop_fit[j] <= old_pop_fit[k] and reverse:
            new_pop.append(old_pop[k])
            new_pop_fit.append(old_pop_fit[k])

        else:
            new_pop.append(old_pop[k])
            new_pop_fit.append(old_pop_fit[k])

    return np.array(new_pop), np.array(new_pop_fit)



def mutx(old_pop, rate, space):

    shape = old_pop.shape

    rate = rate_check(rate)

    muts = math.floor(shape[0] * shape[1] * rate)

    space_dif = np.subtract(space[1], space[0])

    for i in range(muts):
        row = math.ceil(random.random() * shape[0]) - 1
        column = math.ceil(random.random() * shape[1]) - 1

        old_pop[row, column] = mut_check(random.random(
        ) * space_dif[column] + space[0, column], space[0, column], space[1, column])

    return old_pop


def muta(old_pop, rate, amp, space):

    shape = old_pop.shape

    rate = rate_check(rate)

    muts = math.floor(shape[0] * shape[1] * rate)

    for i in range(muts):
        row = math.ceil(random.random() * shape[0]) - 1
        column = math.ceil(random.random() * shape[1]) - 1

        old_pop[row, column] = mut_check(
            old_pop[row, column] + random.uniform(-1, 1) * amp[column], space[0, column], space[1, column])

    return old_pop

def mutm(old_pop, rate, amp, space):

    shape = old_pop.shape

    rate = rate_check(rate)

    muts = math.floor(shape[0] * shape[1] * rate)

    amp_diff = np.subtract(amp[1], amp[0])

    for i in range(muts):
        row = math.ceil(random.random() * shape[0]) - 1
        column = math.ceil(random.random() * shape[1]) - 1

        old_pop[row, column] = mut_check(old_pop[row, column] * (random.random(
        ) * amp_diff[column] + amp[0, column]), space[0, column], space[1, column])

    return old_pop

def swapgen(old_pop, rate):

    shape = old_pop.shape

    rate = rate_check(rate)

    mutations = math.ceil(shape[0] * shape[1] * rate)

    for i in range(mutations):
        r = math.ceil(random.random() * shape[0]) - 1

        c1 = 0
        c2 = 0
        while c1 == c2:
            c1 = math.ceil(random.random() * shape[1]) - 1
            c2 = math.ceil(random.random() * shape[1]) - 1

        old_pop[r, [c1, c2]] = old_pop[r, [c2, c1]]


def swappart(old_pop, rate):

    shape = old_pop.shape

    rate = rate_check(rate)

    mutations = math.ceil(shape[0] * rate)

    for i in range(mutations):
        r = math.ceil(random.random() * shape[0]) - 1
        c = math.ceil(random.random() * shape[1]) - 1
        old_pop[r] = np.concatenate((old_pop[r][c:], old_pop[r][0:c]))



def intmedx(old_pop, alpha, mode):

    shape = old_pop.shape

    pair_list = list(range(shape[0]))
    if mode == 0:
        random.shuffle(pair_list)

    for cyk in range(int(shape[0] / 2)):

        i, j = pair_list[2 * cyk], pair_list[2 * cyk + 1]

        for g in range(shape[1]):

            dx = abs(old_pop[i, g] - old_pop[j, g])

            if old_pop[i, g] < old_pop[j, g]:
                old_pop[i, g] += random.random() * alpha * dx
                old_pop[j, g] -= random.random() * alpha * dx
            else:
                old_pop[i, g] -= random.random() * alpha * dx
                old_pop[j, g] += random.random() * alpha * dx


def crossgrp(old_pop, n):

    shape = old_pop.shape
    new_pop = []

    for row in range(n):

        tmp = []

        for col in range(shape[1]):

            m = math.ceil(random.random() * shape[0]) - 1
            tmp.append(old_pop[m, col])

        new_pop.append(tmp)

    return np.array(new_pop)


def crossov(old_pop, pts, mode):

    shape = old_pop.shape

    pair_list = list(range(shape[0]))
    if mode == 0:
        random.shuffle(pair_list)

    for cyk in range(int(shape[0] / 2)):
        i, j = pair_list[2 * cyk], pair_list[2 * cyk + 1]

        split_points_list = [
            x + 1 for x in sorted(random.sample(range(shape[1] - 2), pts))]
        split_points_list.insert(0, 0)
        split_points_list.append(shape[1])

        old_pop[i], old_pop[j] = splitting(old_pop, split_points_list, i, j, 1)

    return old_pop



def genrpop(pop_size, space):

    shape = space.shape
    new_pop = []

    for row in range(pop_size):
        lst = []

        for column in range(shape[1]):
            limit = space[1][column] - space[0][column]
            lst.append(random.random() * limit + space[0][column])
            lst = pop_check(lst, column, space)

        new_pop.append(lst)

    return np.array(new_pop)


def genrpop_perm(pop_size, min_index, max_index):
    assert max_index >= min_index, "max_index should be greater than min_index"
    return np.random.rand(pop_size, max_index-min_index+1).argsort(axis=1) + min_index

def invfit(old):

    new = (max(old) - old) + min(old)

    return new


def schwefel(pop):

    shape = pop.shape
    fit = np.zeros(shape[0])

    for i in range(shape[0]):
        fit[i] -= np.sum(pop[i] * np.sin(np.sqrt(np.abs(pop[i]))))

    return fit


def eggholder(pop):

    shape = pop.shape
    fit = np.zeros(shape[0])

    for i in range(shape[0]):
        x = pop[i]
        for j in range(shape[1] - 1):
            fit[i] -= x[j] * math.sin(math.sqrt(abs(x[j] - (x[j + 1] + 47)))) + (
                x[j + 1] + 47) * math.sin(math.sqrt(abs(x[j + 1] + 47 + x[j] / 2)))

    return fit


def rastrigin(pop):

    shape = pop.shape

    fit = np.ones(shape[0]) * 10 * shape[1]

    pop_array = np.array(pop)

    for i in range(shape[0]):
        fit[i] += np.sum(pop_array[i]**2 - 10 *
                         np.cos(2 * np.pi * pop_array[i]))

    return fit



def output(pop):
    for tmp in pop:
        print(str(["{0:0.2f}".format(i) for i in tmp]).replace("'", ""))
    print()


def rate_check(rate):
    if rate > 1:
        rate = 1

    elif rate < 0:
        rate = 0

    return rate


def pop_check(pop, column, space):
    if pop[column] < space[0][column]:
        pop[column] = space[0][column]

    if pop[column] > space[1][column]:
        pop[column] = space[1][column]

    return pop


def mut_check(d, space_l, space_u):

    if d < space_l:
        d = space_l

    elif d > space_u:
        d = space_u

    return d


def splitting(old_pop, pts_list, i, j, mode):
    counter = 0
    new_row1 = np.array([])
    new_row2 = np.array([])

    for cyk in range(len(pts_list) - 1):
        if mode == 1:
            if counter % 2 == 0:
                row_index1 = i
                row_index2 = j
            else:
                row_index1 = j
                row_index2 = i
        elif mode == 2:
            if counter % 2 == 0:
                row_index1 = j
                row_index2 = i
            else:
                row_index1 = i
                row_index2 = j

        counter += 1

        new_row1 = np.append(
            new_row1, old_pop[row_index1, pts_list[cyk]:pts_list[cyk + 1]])
        new_row2 = np.append(
            new_row2, old_pop[row_index2, pts_list[cyk]:pts_list[cyk + 1]])

    return new_row1, new_row2

def migrate(pops,popd, i_dest):
    popd[-i_dest,:] = pops[0,:]
    return popd

def warming(old_pop, rate, space):
    dist = space*rate
    s_pop = old_pop.shape
    d_pop = np.array([])
    r_dist = np.array([])

    for j in len(dist):
        r_dist[j] = dist(random.randintint(0,1))

    for i in s_pop[0]:
        t_pop = old_pop[i,:] + r_dist
        for j in s_pop[1]:
            if (t_pop[j] > space[1,j]):
                 t_pop[j] = space[1,j]
            if (t_pop[j] < space[0,j]):
                t_pop[j] = space[0,j]
        d_pop[i,:] = t_pop
    return d_pop

def reset(oldpop, lpop, space):
    return genrpop(lpop, space)
