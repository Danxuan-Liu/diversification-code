import numpy as np
from random import randint,random
import math

global H_feature,H_label,H_featureS,H_feature_lable
H_feature={}
H_label={}
H_feature_lable={}
H_featureS={}

global feature_item,label_num
feature_item=68
label_num=174


def readNMI(address,row,col):
    f = open(address)
    nmi = np.zeros((row, col))
    text=f.read()
    lines=text.split("\n")
    i,j=0,0
    for line in lines:
        tokens=line.split(" ")
        for v in range(len(tokens)-1):
            nmi[i][j]=float(tokens[v])
            j+=1
        i+=1
        j=0
    return nmi
def readDis(address,n):
    f = open(address)
    dis = np.zeros((n, n))
    text = f.read()
    lines = text.split("\n")
    i, j = 0, 0
    for line in lines:
        j=i+1
        tokens = line.split(" ")
        for v in range(len(tokens) - 1):
            dis[i][j] = float(tokens[v])
            j += 1
        i += 1
    for i in range(1,n):
        for j in range(i):
            dis[i][j]=dis[j][i]
    return dis

class GSEMO:
    def __init__(self,**kwargs):
        self.k = kwargs["k"]
        self.n = kwargs["n"]
        self.mylambda = kwargs["mylambda"]
        self.top = kwargs["top"]
        self.l = kwargs["l"]
        iterationoTime = math.exp(1) * self.n * self.k * self.k * self.k / 2
        self.iterationTime = iterationoTime

        self.NMI = readNMI(kwargs["NMI"], kwargs["n"], kwargs["l"])
        self.dis = readDis(kwargs["dis"], kwargs["n"])

    def mutation(self, s):
        rand_rate = 1.0 / (self.n)
        change = np.random.binomial(1, rand_rate, self.n)
        return np.abs(s - change)

    def doGSEMO(self, path):
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        self.tempOptimum = []
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count
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

    def sum_of_top(self,linklist,size,l):
        values = []
        for i in range(size):
            values.append(self.NMI[linklist[i][1]][l])
        values.sort(reverse=True)
        value = 0
        for i in range(min(self.top, size)):
            value += values[i]
        return value

    def metric(self,linklist,i,j):
        return self.dis[linklist[i][1]][linklist[j][1]]



    def evaluateObjective(self,offSpring):
        index=np.nonzero(offSpring)
        size=np.shape(index)[1]

        linklist=[]
        for i, j in zip(index[0], index[1]):
            linklist.append([i, j])
        g=0
        for l in range(self.l):
            g+= self.sum_of_top(linklist,size,l)

        div=0
        for i in range(size):
            for j in range(i+1,size):
                div+=self.metric(linklist,i,j)

        res=0.5*(1+size/self.k)*g+self.mylambda*div
        return res


    def Calucalate_true_value(self,res):
        index = np.nonzero(res)
        size = np.shape(index)[1]

        linklist = []
        for i, j in zip(index[0], index[1]):
            linklist.append([i, j])
        g = 0
        for l in range(self.l):
            g += self.sum_of_top(linklist, size, l)

        div = 0
        for i in range(size):
            for j in range(i + 1, size):
                div += self.metric(linklist, i, j)

        res =  g + self.mylambda * div
        return res