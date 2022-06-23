"""
    Genetic algorithm
"""
import numpy as np
import numpy.random as random
from matplotlib import pyplot as plt

class GA():
    def __init__(self, selection_method="Elite", mutation_rate=0., crossover_ratio=0.3, num_answer=4, num_individuals=10, num_genes=10):
        self.selection_method = selection_method    # Elite or Rulet
        self.mutation_rate = mutation_rate
        self.num_individuals = num_individuals
        self.crossover_ratio = int(crossover_ratio * num_individuals)
        self.answer = random.randint(num_answer, size=num_genes)+1
        self.generation = 1
        self.populations = random.randint(num_answer, size=(num_individuals, num_genes)) + 1
        self.num_answer = num_answer
        self.num_genes = num_genes

    def get_populations(self):
        return self.populations

    def set_populations(self, populations):
        self.populations = populations

    def eval_fitness(self, populations):
        fitness = np.zeros(self.num_individuals)
        for i, individual in enumerate(populations):
            fitness[i] = len(individual[individual - self.answer == 0]) / self.num_genes
        return fitness

    def isStop(self, fitness):
        # Best fitness
        print("*** Top fitness of", self.generation, "Generation : ", np.sort(fitness)[-1])
        if 1.0 in fitness:
            return True
        return False

    def sort_popluations(self, populations, fitness):
        sorted_fitness_arg = np.argsort(-fitness)
        sorted_populations = populations[sorted_fitness_arg]    # Sorted as descending for fitness.
        return sorted_populations

    def selection_and_crossover(self, populations, fitness):
        # Elite
        if self.selection_method == "Elite":
            next_gen_popluations = np.zeros((self.num_individuals, self.num_genes))
            # crossover_ratio 만큼은 다음 세대로 그대로 간다.
            for i in range(self.num_individuals - self.crossover_ratio):
                next_gen_popluations[i] = populations[i]
            # 나머지끼리 crossover 하여 다음세대 자식을 생성.
            for i in range(self.num_individuals - self.crossover_ratio, self.num_individuals):
                pair = random.randint(self.num_individuals, size=2)
                for j in range(self.num_genes):
                    gene_probability = random.randint(100)+1
                    if gene_probability < self.mutation_rate * 10 :         # mutation occur. (Only for child not for parents)
                        next_gen_popluations[i][j] = random.randint(self.num_answer)+1
                    elif gene_probability < 50 + self.mutation_rate * 5:    # father
                        next_gen_popluations[i][j] = populations[pair[0]][j]
                    else:                                                   # mother
                        next_gen_popluations[i][j] = populations[pair[1]][j]
            return next_gen_popluations
        if self.selection_method == "Rulet":
            next_gen_popluations = np.zeros((self.num_individuals, self.num_genes))
            # fitness 확률에 비례하여 다음 세대로 갈 probability 가 결정된다.
            for i in range(self.num_individuals):
                if random.randint(100) * 0.01 < fitness[i]:
                    next_gen_popluations[i] = populations[i]
                # Else crossover.
                else:
                    pair = random.randint(self.num_individuals, size=2)
                    for j in range(self.num_genes):
                        gene_probability = random.randint(100)+1
                        if gene_probability < self.mutation_rate * 10 :         # mutation occur.
                            next_gen_popluations[i][j] = random.randint(self.num_answer)+1
                        elif gene_probability < 50 + self.mutation_rate * 5:    # father
                            next_gen_popluations[i][j] = populations[pair[0]][j]
                        else:                                                   # mother
                            next_gen_popluations[i][j] = populations[pair[1]][j]
            return next_gen_popluations


ga = GA("Elite", mutation_rate=0.2, crossover_ratio=0.7, num_answer=3, num_individuals=10, num_genes=20)
while(1):
    current_gen_population = ga.get_populations()
    fitness = ga.eval_fitness(current_gen_population)
    if(ga.isStop(fitness)):
        break
    if ga.selection_method == "Elite":
        sorted_populations = ga.sort_popluations(ga.get_populations(), fitness)
        next_gen_popluations = ga.selection_and_crossover(sorted_populations, fitness)

    if ga.selection_method == "Rulet":
        next_gen_popluations = ga.selection_and_crossover(current_gen_population, fitness)
    ga.set_populations(next_gen_popluations)
    ga.generation += 1









fitness_list = []
generation_list = []
results = 0
for i in range(20):
    ga = GA("Rulet", mutation_rate=0.4, num_answer=4, num_individuals=10, num_genes=20)
    # ga = GA("Elite", mutation_rate=0.2, crossover_ratio=0.7, num_answer=3, num_individuals=10, num_genes=20)
    while(1):
        current_gen_population = ga.get_populations()
        fitness = ga.eval_fitness(current_gen_population)
        if(ga.isStop(fitness)):
            break
        if ga.selection_method == "Elite":
            sorted_populations = ga.sort_popluations(ga.get_populations(), fitness)
            next_gen_popluations = ga.selection_and_crossover(sorted_populations, fitness)

        if ga.selection_method == "Rulet":
            next_gen_popluations = ga.selection_and_crossover(current_gen_population, fitness)
        ga.set_populations(next_gen_popluations)
        fitness_list.append(np.average(fitness))
        generation_list.append(ga.generation)
        ga.generation += 1
print("results : ", results)
print("Average generation of goal : ", results/20.0)

fitness_array = np.array(fitness_list)
generation_array = np.array(generation_list)

plt.plot(generation_array, fitness_array, 'r')

plt.show()