import sys
import math
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import animation
import copy
import random


data = []


# plan is an array of 40 floating point numbers
def sim(plan):
    for i in range(0, len(plan)):
        if plan[i] > 1:
            plan[i] = 1.0
        elif plan[i] < -1:
            plan[i] = -1.0

    dt = 0.1
    friction = 1.0
    gravity = 0.1
    mass = [30, 10, 5, 10, 5, 10]
    edgel = [0.5, 0.5, 0.5, 0.5, 0.9]
    edgesp = [160.0, 180.0, 160.0, 180.0, 160.0]
    edgef = [8.0, 8.0, 8.0, 8.0, 8.0]
    anglessp = [20.0, 20.0, 10.0, 10.0]
    anglesf = [8.0, 8.0, 4.0, 4.0]

    edge = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5)]
    angles = [(4, 0), (4, 2), (0, 1), (2, 3)]

    # vel and pos of the body parts, 0 is hip, 5 is head, others are joints
    v = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    p = [[0, 0, -0.25, 0.25, 0.25, 0.15], [1, 0.5, 0, 0.5, 0, 1.9]]

    spin = 0.0
    maxspin = 0.0
    lastang = 0.0

    for j in range(20):
        for k in range(10):
            lamb = 0.05 + 0.1 * k
            t0 = 0.5
            if j > 0:
                t0 = plan[2 * j - 2]
            t0 *= 1 - lamb
            t0 += plan[2 * j] * lamb

            t1 = 0.0
            if j > 0:
                t1 = plan[2 * j - 1]
            t1 *= 1 - lamb
            t1 += plan[2 * j + 1] * lamb

            contact = [False, False, False, False, False, False]
            for z in range(6):
                if p[1][z] <= 0:
                    contact[z] = True
                    spin = 0
                    p[1][z] = 0

            anglesl = [-(2.8 + t0), -(2.8 - t0), -(1 - t1) * 0.9, -(1 + t1) * 0.9]

            disp = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            dist = [0, 0, 0, 0, 0]
            dispn = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            for z in range(5):
                disp[0][z] = p[0][edge[z][1]] - p[0][edge[z][0]]
                disp[1][z] = p[1][edge[z][1]] - p[1][edge[z][0]]
                dist[z] = (
                    math.sqrt(disp[0][z] * disp[0][z] + disp[1][z] * disp[1][z]) + 0.01
                )
                inv = 1.0 / dist[z]
                dispn[0][z] = disp[0][z] * inv
                dispn[1][z] = disp[1][z] * inv

            dispv = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            distv = [0, 0, 0, 0, 0]
            for z in range(5):
                dispv[0][z] = v[0][edge[z][1]] - v[0][edge[z][0]]
                dispv[1][z] = v[1][edge[z][1]] - v[1][edge[z][0]]
                distv[z] = 2 * (disp[0][z] * dispv[0][z] + disp[1][z] * dispv[1][z])

            forceedge = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            for z in range(5):
                c = (edgel[z] - dist[z]) * edgesp[z] - distv[z] * edgef[z]
                forceedge[0][z] = c * dispn[0][z]
                forceedge[1][z] = c * dispn[1][z]

            edgeang = [0, 0, 0, 0, 0]
            edgeangv = [0, 0, 0, 0, 0]
            for z in range(5):
                edgeang[z] = math.atan2(disp[1][z], disp[0][z])
                edgeangv[z] = (dispv[0][z] * disp[1][z] - dispv[1][z] * disp[0][z]) / (
                    dist[z] * dist[z]
                )

            inc = edgeang[4] - lastang
            if inc < -math.pi:
                inc += 2.0 * math.pi
            elif inc > math.pi:
                inc -= 2.0 * math.pi
            spin += inc
            spinc = spin - 0.005 * (k + 10 * j)
            if spinc > maxspin:
                maxspin = spinc
                lastang = edgeang[4]

            angv = [0, 0, 0, 0]
            for z in range(4):
                angv[z] = edgeangv[angles[z][1]] - edgeangv[angles[z][0]]

            angf = [0, 0, 0, 0]
            for z in range(4):
                ang = edgeang[angles[z][1]] - edgeang[angles[z][0]] - anglesl[z]
                if ang > math.pi:
                    ang -= 2 * math.pi
                elif ang < -math.pi:
                    ang += 2 * math.pi
                m0 = dist[angles[z][0]] / edgel[angles[z][0]]
                m1 = dist[angles[z][1]] / edgel[angles[z][1]]
                angf[z] = ang * anglessp[z] - angv[z] * anglesf[z] * min(m0, m1)

            edgetorque = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            for z in range(5):
                inv = 1.0 / (dist[z] * dist[z])
                edgetorque[0][z] = -disp[1][z] * inv
                edgetorque[1][z] = disp[0][z] * inv

            for z in range(4):
                i0 = angles[z][0]
                i1 = angles[z][1]
                forceedge[0][i0] += angf[z] * edgetorque[0][i0]
                forceedge[1][i0] += angf[z] * edgetorque[1][i0]
                forceedge[0][i1] -= angf[z] * edgetorque[0][i1]
                forceedge[1][i1] -= angf[z] * edgetorque[1][i1]

            f = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            for z in range(5):
                i0 = edge[z][0]
                i1 = edge[z][1]
                f[0][i0] -= forceedge[0][z]
                f[1][i0] -= forceedge[1][z]
                f[0][i1] += forceedge[0][z]
                f[1][i1] += forceedge[1][z]

            for z in range(6):
                f[1][z] -= gravity * mass[z]
                invm = 1.0 / mass[z]
                v[0][z] += f[0][z] * dt * invm
                v[1][z] += f[1][z] * dt * invm

                if contact[z]:
                    fric = 0.0
                    if v[1][z] < 0.0:
                        fric = -v[1][z]
                        v[1][z] = 0.0

                    s = np.sign(v[0][z])
                    if v[0][z] * s < fric * friction:
                        v[0][z] = 0
                    else:
                        v[0][z] -= fric * friction * s
                p[0][z] += v[0][z] * dt
                p[1][z] += v[1][z] * dt

            data.append(copy.deepcopy(p))

            if contact[0] or contact[5]:
                return p[0][5]
    return p[0][5]


###########
# The following code is given as an example to store a video of the run and to display
# the run in a graphics window. You will treat sim(plan) as a black box objective
# function and maximize it.
###########


def init():
    ax.add_patch(patch)
    ax.add_patch(head)
    return patch, head


def animate(j):
    first = []
    second = []
    for i in joints:
        first.append(data[j][0][i])
        second.append(data[j][1][i])
    a = np.array([first, second])
    a = np.transpose(a)
    patch.set_xy(a)
    head.center = (data[j][0][5], data[j][1][5])
    return patch, head


def tournament_select(fitness_list, tournament_probability, tournament_size):
    most_fit = 0
    fitness_index = 0
    individual_index = 1

    population_size = len(fitness_list)
    tour_fitness_and_individual = np.zeros((tournament_size, 2))

    for i in range(tournament_size):
        i_tmp = np.random.randint(0, population_size)
        tour_fitness_and_individual[i, fitness_index] = fitness_list[i_tmp]
        tour_fitness_and_individual[i, individual_index] = i_tmp

    tour_fitness_and_individual = tour_fitness_and_individual[tour_fitness_and_individual[:, fitness_index].argsort()[::-1]]

    while True:
        r = np.random.rand()
        if r < tournament_probability:
            selected_individual_index = int(tour_fitness_and_individual[most_fit, individual_index])
            return selected_individual_index
        else:
            remaining_participants = tour_fitness_and_individual.shape[0]
            if remaining_participants > 1:
                tour_fitness_and_individual = np.delete(tour_fitness_and_individual, most_fit, axis=0)
            else:
                selected_individual_index = int(tour_fitness_and_individual[most_fit, individual_index])
                return selected_individual_index


def crossover(individual1, individual2):
    n_genes = len(individual1)
    crossover_point = np.random.randint(1, n_genes)
    new_individuals = np.zeros((2, n_genes))

    for j in range(n_genes):
        if j < crossover_point:
            new_individuals[0, j] = individual1[j]
            new_individuals[1, j] = individual2[j]
        else:
            new_individuals[0, j] = individual2[j]
            new_individuals[1, j] = individual1[j]

    return new_individuals


def decode_chromosome(chromosome, number_of_variables, maximum_variable_value):
    x = np.zeros(number_of_variables)
    n_genes = len(chromosome)
    variable_length = int(np.floor(n_genes / number_of_variables))
    i_gene = 0
    maximum_variable_value = np.abs(maximum_variable_value)

    for i in range(number_of_variables):
        x[i] = 0.0
        for j in range(0, variable_length):
            x[i] += 2**-j * chromosome[i_gene]
            i_gene += 1

        # x[i] = -maximum_variable_value + ((2 * maximum_variable_value * x[i]) / (1 - 2**(-variable_length)))
        x[i] = -maximum_variable_value + (2 * maximum_variable_value * x[i]) / (2**variable_length - 1)

    return x


def evaluate_individual(x): # objective funtion
    fitness = sim(x)
    return fitness


def initialize_population(population_size, number_of_genes):
    population = np.zeros((population_size, number_of_genes), dtype=int)

    for i in range(population_size):
        for j in range(number_of_genes):
            r = np.random.rand()
            if r < 0.5:
                population[i, j] = 0
            else:
                population[i, j] = 1

    return population


def mutate(individual, mutation_probability):
    n_genes = len(individual)
    mutated_individual = np.copy(individual)

    for i in range(n_genes):
        r = np.random.rand()
        if r < mutation_probability:
            mutated_individual[i] = 1 - individual[i]

    return mutated_individual


def run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value,
                              tournament_size, tournament_probability, crossover_probability, mutation_probability,
                              number_of_generations):
    maximum_fitness_best = 0
    best_variable_values_best = np.zeros(number_of_variables)
    population = initialize_population(population_size, number_of_genes)

    for generation in range(number_of_generations):
        maximum_fitness = 0.0
        fitness_list = np.zeros(population_size)
        for i in range(population_size):
            chromosome = population[i, :]
            variable_values = decode_chromosome(chromosome, number_of_variables, maximum_variable_value)
            fitness_list[i] = evaluate_individual(variable_values)
            if fitness_list[i] > maximum_fitness:
                maximum_fitness = fitness_list[i]
                i_best_individual = i
                best_variable_values = variable_values
            
            if fitness_list[i] > maximum_fitness_best:
                maximum_fitness_best = fitness_list[i]
                best_variable_values_best = variable_values

        temporary_population = population
        for i in range(0, population_size, 2):
            i1 = tournament_select(fitness_list, tournament_probability, tournament_size)
            i2 = tournament_select(fitness_list, tournament_probability, tournament_size)
            r = np.random.rand()
            if r < crossover_probability:
                individual1 = population[i1, :]
                individual2 = population[i2, :]
                new_individual_pair = crossover(individual1, individual2)
                temporary_population[i, :] = new_individual_pair[0, :]
                temporary_population[i+1, :] = new_individual_pair[1, :]
            else:
                temporary_population[i, :] = population[i1, :]
                temporary_population[i+1, :] = population[i2, :]

        temporary_population[0, :] = population[i_best_individual, :]
        for i in range(1, population_size):
            temp_individual = mutate(temporary_population[i, :], mutation_probability)
            temporary_population[i, :] = temp_individual

        population = temporary_population
        print('generation:', generation, 'maximum_fitness:', maximum_fitness_best, end='\r')

    return maximum_fitness_best, best_variable_values_best


def approx_grad(plan, eta):
    grad = (sim(plan + eta) - sim(plan)) / eta
    return grad


def grad_descent(plan, eta, n_its):
    tot_dists = [0]
    best_plan = plan.copy()

    for t in range(n_its):

        grad = approx_grad(plan, eta)
        alpha = eta
        plan = plan - alpha*grad
        dist = sim(plan)

        if tot_dists[-1] < dist:
            tot_dists.append(dist)
            best_plan = plan.copy()
        else:
            tot_dists.append(tot_dists[-1])

        print('t =', t, 'max dist:', np.max(tot_dists), end='\r') if t%10 == 0 else None

    return best_plan, tot_dists


def random_search(n_its):
    tot_dists = [0]
    best_plan = None

    for t in range(n_its):

        plan = np.random.uniform(-1, 1, 40)
        dist = sim(plan)

        if tot_dists[-1] < dist:
            tot_dists.append(dist)
            best_plan = plan.copy()
        else:
            tot_dists.append(tot_dists[-1])

        print('t =', t, 'max dist:', np.max(tot_dists), end='\r') if t%10 == 0 else None
    
    return best_plan, tot_dists


if __name__ == "__main__":

    n_its = 1000
    eta = 0.1
    samples = np.linspace(0, n_its, n_its+1)
    plan = np.random.uniform(-1, 1, 40)

    best_plan, tot_dists = random_search(n_its)
    print('\nRandom search: \nbest dist:', tot_dists[-1], '\nbest plan:', best_plan)
    best_plan, tot_dists = grad_descent(plan, eta, n_its)
    print('\nGradient descent: \nbest dist:', tot_dists[-1], '\nbest plan:', best_plan)

    population_size = 500
    maximum_variable_value = 1
    number_of_genes = 50
    number_of_variables = 40

    tournament_size = 3
    tournament_probability = 0.785
    crossover_probability = 0.8
    mutation_probability = 0.02
    number_of_generations = 100

    maximum_fitness, best_plan = run_function_optimization(
        population_size, number_of_genes, number_of_variables, maximum_variable_value,
        tournament_size, tournament_probability, crossover_probability, mutation_probability,
        number_of_generations
    )
    print('\nEvolutionary algorithm: \nbest dist:', maximum_fitness, '\nbest plan:', best_plan)

    best_plan = [-1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1]

    dist = sim(best_plan)
    print('Best dist;', dist)
    print('Best plan:', best_plan)

    # draw the simulation
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(12, 3)

    ax = plt.axes(xlim=(-1, 10), ylim=(0, 3))

    joints = [5, 0, 1, 2, 1, 0, 3, 4]
    patch = plt.Polygon([[0, 0], [0, 0]], closed=None, fill=None, edgecolor="k")
    head = plt.Circle((0, 0), radius=0.15, fc="k", ec="k")

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(data), interval=20, repeat=False
    )
    anim.save('animation.gif', fps=50)
    plt.show()