"""
For Creating the Problem
"""
import random   # to generate random num
import numpy as np  # for random choice with probability

"""
Three constaints
1. no two queens can share a same row
2. no two queens can share a same col
3. no two queens can share a same diameter

fitness function = number of nonattacking queens
not what if initialize population we choose greedy instead of random i
greedy = iterates through colun and locate each queen on the row with 
least conflicts
"""

"""
1. Define Cost function, cost, variables, select GA parameters
    here cost = value for fitness
2. generate initial population
    gene = number from 0 to n-1
    chromosomes = array of genes
    population = in each generation has determined the number of chromosomes
    we have to create a random initial population by randomly selecting chromosomes
    the # of generations needed for convergence depends on this random initial pop
3. decode chromosomes
    
    
4. find cost for each chromosome
    to find the assigned cost for each chromosome, a cost function is defind
    the result of the cost function called is called cost value
    -the average of cost values of each generation converges to the desired answer
5. select mates
    chromosomes with a higher fitness(lesser cost) are used to produce the next gen
    the offsprings are a product of the father and mother (crossing over)
6. mating
7. mutation
8. convergence check, if fail, go back to step 3.
9. done
"""
# Variables

chessboardLength = 4    # how many rows/cols in the chessboard
population = 4  # of chromosomes
#iteration
#mutationRate
chessboardMatrix = [[0]*chessboardLength for i in range(chessboardLength)]    # chessboard matrix initialized to 0
chromosomeMatrix = [[0]*chessboardLength for j in range(population)]   # matrix of num from 0 to chessboardLength-1
# the first number defines the place of the queen in the first row and so on

fitnessMatrix = [0 for k in range(population)]   # #for each chromosome matrix a cost value is defined, best is 0
# and when no queen can take the other one

probabilityMatrix = [0 for l in range(population)]  # for each population we have the probability
newGenMatrix = []
crossoverMatrix = [[] for m in range(population)] #for generating children from parents, decides which gene
# to select


# Set the chessboard with 0s
def clear_chessboard():
    for i in range(chessboardLength):
        chessboardMatrix[i] = [0]*chessboardLength


# Print chessboard
def print_chessboard():
    for i in range(chessboardLength):
        print(chessboardMatrix[i])


# Generate the Initial Population by randomly assigning distinct
def initialize_population(matrix):
    for i in range(population):
        for j in range(population):
            matrix[i][j] = j
    random.shuffle([random.shuffle(i) for i in matrix])


# Fill the Chessboard with a chromosome
def fill_chessboard(selection, matrix):
    clear_chessboard()
    for i in range(chessboardLength):
        chessboardMatrix[i][matrix[selection][i]] = 1


# Determine the Cost Value for Each ChromosomeMatrix and keep it in cost matrix
# this will count the # of queens that are hitting each other.
# this is our fitness
# if two queens hit each other, it will be 2,
def calculate_fitness(matrix):

    for selection in range(population):
        fitness = 0
        for i in range(population):
            for j in range(population):
                if i != j:
                    dx = abs(i-j)
                    dy = abs(matrix[selection][i] - matrix[selection][j])
                    if dx == dy:
                        fitness += 1
        fitnessMatrix[selection] = population*(population-1) - fitness

#each diagnoal has n-1*n clashes for 8x8 = 7*8 = 56 clashes, but divide by 2 = 28 to not double count
####the problem here is that fitness is better when lower, but we want better when higher?


# find the probability of selection of string which is fitness/totalfitnes
def calculate_probability():
    for i in range(population):
        probabilityMatrix[i] = fitnessMatrix[i]/sum(fitnessMatrix)


# roulette wheel selection
def roulette_wheel_selection():
    choice = np.random.choice([i for i in range(population)], size=population, p=probabilityMatrix)
    # note, we are given an array of selections
    return choice


# Generate new Initial Population
def new_generation():
    choice = roulette_wheel_selection()
    # remove after test
    print('\nChoice')
    print(choice)
    for i in choice:
        newGenMatrix.append(chromosomeMatrix[i])


# CrossOver
def crossover_population():
    # randomly generate crossover point from 0 to n-1
    crossover_point = random.randint(0, chessboardLength-1)
    #for tests
    print('\nCROSSOVER POINT')
    print(crossover_point)
    #just for tests
    print('\ncrossover')
    for i in range(population):
        select_org_matrix = random.randint(0, chessboardLength - 1)
        for j in range(crossover_point+1):
            #append the newGen's first cross_overpoint then append with the remaining of the random original
            crossoverMatrix[i].append(newGenMatrix[i][j])
        for k in range(crossover_point+1, chessboardLength):
            crossoverMatrix[i].append(chromosomeMatrix[select_org_matrix][k])


# rewrite the functions so taht we pass a matrix to operate on
# cross over matrix is our population after crossover
# now have a new population with mutations
# then find teh fitness of the new population after mutation


# Mutation
def mutation_population(mutation_probability,matrix):
    mp = mutation_probability
    for i in matrix:
        for j in range(chessboardLength):
            temp = j
            matrix[i][j] = np.random.choice([temp, np.random.randint(0, chessboardLength-1)],
                                             p=[1-mp),mp])
#apply mutation probability...


#maybe end after 100 years ? 100 generations
# when n is 100 there should not be a solution.? just supposed to be good

#from the probability randomly get actual count via roulette wheel
# TESTING
print_chessboard()
print('\n')
initialize_population(chromosomeMatrix)
fill_chessboard(0,chromosomeMatrix)

print(chromosomeMatrix)
print('\n')

print_chessboard()
print('\nFitnessMatrix')
calculate_fitness(chromosomeMatrix)
print(fitnessMatrix)
calculate_probability()
print('\nProbMatrix')
print(probabilityMatrix)
new_generation()
print(newGenMatrix)
crossover_population()
print(crossoverMatrix)
mutation_population(.001,crossoverMatrix)
