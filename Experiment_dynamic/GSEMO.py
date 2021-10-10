import numpy as np
from random import randint,random
class GSEMO:

    def __init__(self,**kwargs):
        self.constraint= kwargs["k"]
        self.n = kwargs["n"]
        self.weight=kwargs["w"]
        self.distance=kwargs["dis"]
        self.mylambda=kwargs["mylambda"]
        self.iteration=kwargs["iteration"]

    def setIterationTime(self,time):
        self.iterationTime=time

    def mutation(self, s):
        rand_rate = 1.0 / (self.n)
        change = np.random.binomial(1, rand_rate, self.n)
        return np.abs(s - change)

    def setInilSolution(self,init_solution):
        self.population=np.mat(init_solution)
        self.fitness = np.mat(np.zeros([1, 2]))
        # update the size of greedy algorithm
        self.fitness[0, 1] = self.population[0, :].sum()
        # update the true value of greedy algorithm
        self.fitness[0, 0] = self.evaluateObjective(self.population)
    def updateFitness(self):
        popSize = self.population.shape[0]
        for i in range(popSize):
            self.fitness[i,0]=self.evaluateObjective(self.population[i])#
    def GSEMO(self):
        self.updateFitness()
        popSize =self.population.shape[0]
        t = 0  # the current iterate countj

        while t < self.iteration:#Iteration termination condition

            s = self.population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1,2])) #value, size

            offSpringFit[0, 1] = offSpring[0, :].sum()
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.constraint:
                t += 1
                continue

            offSpringFit[0, 0]=self.evaluateObjective(offSpring)

            isDominate = False
            for i in range(0, popSize):
                if (self.fitness[i, 0] > offSpringFit[0, 0] and self.fitness[i, 1] <= offSpringFit[0, 1]) or (
                        self.fitness[i, 0] >= offSpringFit[0, 0] and self.fitness[i, 1] < offSpringFit[0, 1]):
                    isDominate = True
                    break
            if isDominate == False:  # there is no better individual than offSpring
                Q = []
                for j in range(0, popSize):
                    if offSpringFit[0, 0] >= self.fitness[j, 0] and offSpringFit[0, 1] <= self.fitness[j, 1]:
                        continue
                    else:
                        Q.append(j)

                self.fitness = np.vstack((offSpringFit, self.fitness[Q, :]))  # update fitness
                self.population = np.vstack((offSpring, self.population[Q, :]))  # update population
            t = t + 1
            popSize = np.shape(self.fitness)[0]

        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if self.fitness[p, 1] <= self.constraint and self.fitness[p, 0] > maxValue:
                maxValue = self.fitness[p, 0]
                resultIndex = p

        return self.population[resultIndex]

    def evaluateObjective(self,offSpring):
        index=np.nonzero(offSpring)
        size=np.shape(index)[1]

        linklist=[]
        for i, j in zip(index[0], index[1]):
            linklist.append([i, j])

        f=0
        div=0
        for i in linklist:
            f+=self.weight[i[1]]
        for i in range(size):
            for j in range(i+1,size):
                div+=self.distance[linklist[i][1]][linklist[j][1]]
        res=0.5*(1+size/self.constraint)*f+self.mylambda*div
        return res

    def change_weight(self,index,w):
        self.weight[index]=w
    def change_distance(self,u,v,w):
        self.distance[u][v]=w
        self.distance[v][u]=w