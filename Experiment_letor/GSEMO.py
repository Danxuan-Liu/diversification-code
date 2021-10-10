import numpy as np
from random import randint,random
class GSEMO:

    def __init__(self,**kwargs):
        self.k= kwargs["k"]
        self.n = kwargs["n"]
        self.weight=kwargs["w"]
        self.distance=kwargs["dis"]
        self.mylambda=kwargs["mylambda"]

    def setIterationTime(self,time):
        self.iterationTime=time

    def mutation(self, s):
        rand_rate = 1.0 / (self.n)
        change = np.random.binomial(1, rand_rate, self.n)
        return np.abs(s - change)

    def Calucalate_true_value(self,res):
        index = np.nonzero(res)
        size = np.shape(index)[1]

        linklist = []
        for i, j in zip(index[0], index[1]):
            linklist.append([i, j])

        f = 0
        div = 0
        for i in linklist:
            # print(self.weight[i[1]])
            f += self.weight[i[1]]
        for i in range(size):
            for j in range(i + 1, size):
                div += self.distance[linklist[i][1]][linklist[j][1]]
        res = f + self.mylambda * div
        return res

    def doGSEMO(self, path):
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        self.tempOptimum = []
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count j
        iter = 0
        kn = int(self.k * self.n)
        while t < self.iterationTime:
            if iter == kn:
                log = open(path, 'a')
                iter = 0
                resultIndex = -1
                maxValue = float("-inf")
                for p in range(0, popSize):
                    if fitness[p, 1] <= self.k and fitness[p, 0] > maxValue:
                        maxValue = fitness[p, 0]
                        resultIndex = p

                self.tempOptimum.append(population[resultIndex])
                res = population[resultIndex]
                f = self.Calucalate_true_value(res)
                log.write(str(f))
                log.write("\n")

                index = np.nonzero(res)

                linklist = []
                for i, j in zip(index[0], index[1]):
                    linklist.append([i, j])

                for item in linklist:
                    log.write(str(item[1] + 1))
                    log.write(' ')
                log.write("\n")
                log.close()

            iter += 1
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly 取某一行
            offSpring = self.mutation(s)  # every bit will be flipped with probability 1/n
            offSpringFit = np.mat(np.zeros([1, 2]))  # value, size

            offSpringFit[0, 1] = offSpring[0, :].sum()
            if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.k:
                t += 1
                continue
            offSpringFit[0, 0] = self.evaluateObjective(offSpring)

            isDominate = False
            for i in range(0, popSize):
                if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                        fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                    isDominate = True
                    break
            if isDominate == False:  # there is no better individual than offSpring
                Q = []
                for j in range(0, popSize):
                    if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                        continue
                    else:
                        Q.append(j)

                fitness = np.vstack((offSpringFit, fitness[Q, :]))  # update fitness
                population = np.vstack((offSpring, population[Q, :]))  # update population
            t = t + 1
            popSize = np.shape(fitness)[0]

        resultIndex = -1
        maxValue = float("-inf")
        for p in range(0, popSize):
            if fitness[p, 1] <= self.k and fitness[p, 0] > maxValue:
                maxValue = fitness[p, 0]
                resultIndex = p

        self.tempOptimum.append(population[resultIndex])
        return self.tempOptimum

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
        res=0.5*(1+size/self.k)*f+self.mylambda*div
        return res
