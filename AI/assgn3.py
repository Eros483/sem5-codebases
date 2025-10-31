import random
import numpy as np
from typing import List, Tuple

class Q1:
    """
    Generational GA as instructed in Q1
    """
    def __init__(self, population: List[str], mutation_prob: float = 0.25):
        self.population=population
        self.mutation_prob=mutation_prob
        self.pop_size=len(population)
        self.chromosome_length=len(population[0])
        
    def fitness(self, binary_str: str) -> float:
        """
        Provided fitness function: 15x-x^2
        """
        x=int(binary_str, 2)
        return 15*x - x**2
    
    def proportionate_selection(self, fitness_values: List[float]) -> List[str]:
        """
        Method of selection: Proportionate selection
        i.e Roulette wheel selection
        """
        total_fitness=sum(fitness_values)
        if total_fitness==0:
            return random.choices(self.population, k=self.pop_size)

        min_fitness=min(fitness_values)
        if min_fitness<0:
            adjusted_fitness=[f - min_fitness + 1 for f in fitness_values]
        else:
            adjusted_fitness = fitness_values
        
        total=sum(adjusted_fitness)
        probabilities=[f/total for f in adjusted_fitness]
        
        selected=random.choices(self.population, weights=probabilities, k=self.pop_size)
        return selected
    
    def single_point_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        Provided method of crossover
        Single point crossover
        """
        point= random.randint(1, self.chromosome_length-1)
        child1= parent1[:point] + parent2[point:]
        child2= parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, chromosome: str) -> str:
        """
        Bit flip mutation with probability 0.25
        """
        mutated=list(chromosome)
        for i in range(len(mutated)):
            if random.random()<self.mutation_prob:
                mutated[i]='1' if mutated[i] == '0' else '0'
        return ''.join(mutated)
    
    def evolve(self, iterations: int = 100):
        """
        Run the generational GA algorithm for 100 iterations base
        """
        best_fitness_history=[]
        avg_fitness_history=[]
        
        for iteration in range(iterations):
            fitness_values = [self.fitness(chrom) for chrom in self.population]
            
            #Recording statistics at each iteration
            best_fitness = max(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            selected = self.proportionate_selection(fitness_values)

            new_population = []
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    child1, child2 = self.single_point_crossover(selected[i], selected[i+1])
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            new_population = [self.mutate(chrom) for chrom in new_population]
            self.population = new_population[:self.pop_size]
        
        #Statistics after 100 iterations
        final_fitness = [self.fitness(chrom) for chrom in self.population]
        best_idx = final_fitness.index(max(final_fitness))
        best_chromosome = self.population[best_idx]
        best_x = int(best_chromosome, 2)
        
        return {
            'best_chromosome': best_chromosome,
            'best_x': best_x,
            'best_fitness': max(final_fitness),
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'final_population': self.population
        }

class Q2:
    """
    Algorithm for GA as instructed in Question 2 
    """
    def __init__(self, population: List[str], mutation_prob: float = 0.1):
        self.population=population
        self.mutation_prob=mutation_prob
        self.pop_size=len(population)
        self.chromosome_length=len(population[0])
        
    def fitness(self, chromosome: str) -> float:
        """
        Fitness function: (a+b)-(c+d)+(e+f)-(g+h) as provided
        """
        genes = [int(g) for g in chromosome]
        return (genes[0] + genes[1]) - (genes[2] + genes[3]) + \
               (genes[4] + genes[5]) - (genes[6] + genes[7])
    
    def proportionate_selection(self) -> str:
        """
        Roulette wheel selection - returns single individual
        similiar to q1
        """
        fitness_values=[self.fitness(chrom) for chrom in self.population]

        min_fitness = min(fitness_values)
        if min_fitness < 0:
            adjusted_fitness = [f - min_fitness + 1 for f in fitness_values]
        else:
            adjusted_fitness = fitness_values
        
        total=sum(adjusted_fitness)
        if total==0:
            return random.choice(self.population)
        
        probabilities =[f / total for f in adjusted_fitness]
        return random.choices(self.population, weights=probabilities, k=1)[0]
    
    def midpoint_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        Crossover method as specified in q2
        Midpoint crossover - split at middle of chromosome
        """
        midpoint = self.chromosome_length // 2
        child1 = parent1[:midpoint] + parent2[midpoint:]
        child2 = parent2[:midpoint] + parent1[midpoint:]
        return child1, child2
    
    def mutate(self, chromosome: str) -> str:
        """
        Random digit mutation as specified in question
        """
        mutated = list(chromosome)
        for i in range(len(mutated)):
            if random.random() < self.mutation_prob:
                mutated[i] = str(random.randint(0, 9))
        return ''.join(mutated)
    
    def evolve(self, iterations: int = 100):
        """
        Run the steady-state GA for 100 iterations
        """
        best_fitness_history = []
        avg_fitness_history = []
        
        for _ in range(iterations):
            fitness_values = [self.fitness(chrom) for chrom in self.population]
            
            #keeping a check of statistics at each iteration
            best_fitness=max(fitness_values)
            avg_fitness=sum(fitness_values) /len(fitness_values)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            parent1 = self.proportionate_selection()
            parent2 = self.proportionate_selection()
        
            child1, child2 = self.midpoint_crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            fitness_with_children = fitness_values + [self.fitness(child1), self.fitness(child2)]
            population_with_children = self.population + [child1, child2]

            sorted_pairs = sorted(zip(fitness_with_children, population_with_children), 
                                key=lambda x: x[0], reverse=True)
            self.population = [chrom for _, chrom in sorted_pairs[:self.pop_size]]
        
        #Statistics at the end of 100 iterations
        final_fitness=[self.fitness(chrom) for chrom in self.population]
        best_idx=final_fitness.index(max(final_fitness))
        best_chromosome=self.population[best_idx]
        
        return {
            'best_chromosome': best_chromosome,
            'best_fitness': max(final_fitness),
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'final_population': self.population
        }

def main():
#-------------------------------------------------------------------------------------------
    print("--------------Q1---------------")

    initial_pop_q1=['1100', '0100', '0001', '1110', '0111', '1001']
    ga_q1 = Q1(initial_pop_q1.copy(), mutation_prob=0.25)
    results_q1 = ga_q1.evolve(iterations=100)

    print(f"Best Chromosome set: {results_q1['best_chromosome']}")
    print(f"Best x: {results_q1['best_x']}")
    print(f"Best Fitness: {results_q1['best_fitness']:.2f}")
    print(f"Final Population: {results_q1['final_population']}")
#-------------------------------------------------------------------------------------------
    print("--------------Q2---------------")

    initial_pop_q2=['65413532', '87126601', '23921285', '41852094']
    ga_q2 = Q2(initial_pop_q2.copy(), mutation_prob=0.1)
    results_q2 = ga_q2.evolve(iterations=100)

    print(f"Best Chromosome set: {results_q2['best_chromosome']}")
    print(f"Best Fitness: {results_q2['best_fitness']:.2f}")
    print(f"Final Population: {results_q2['final_population']}")

if __name__ == "__main__":
    random.seed(42)  #for ensuring consistent results
    main()