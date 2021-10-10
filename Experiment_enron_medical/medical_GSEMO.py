import numpy as np

from random import randint,random

import math

global H_feature,H_label,H_featureS,H_feature_lable
H_feature={}
H_label={}
H_feature_lable={}
H_featureS={}

global feature_item,label_num
feature_item=1449
label_num=45

def read_sparse_arff(f_path, xml_path):
    global feature_item
    f = open(f_path)
    f_data = f.read().split('@data')
    f.close()
    column = [i.split(' ')[1] for i in f_data[0].split('@attribute')[1:]]
    label_row, label_col, label_data, label_indptr = [], [], [], []
    feature_row, feature_col, feature_data, feature_indptr = [], [], [], []
    inil = 0
    inil_label = 0
    for l in enumerate(f_data[1].replace(' ', ':').split('\n')[1:-1]):
        l_v_dict = eval(l[1])
        col, data = [], []
        col.extend(l_v_dict.keys())
        data.extend(l_v_dict.values())
        index = 0
        feature_num = 0
        for item in col:
            if item < feature_item:
                index += 1
                feature_num += 1
            else:
                break
        inil_label += len(l_v_dict) - feature_num
        label_indptr.append(inil_label)
        label_col.extend(col[index:])
        label_data.extend(data[index:])
        label_row.extend([l[0] for i in range(len(l_v_dict) - feature_num)])

        inil += feature_num
        feature_indptr.append(inil)
        feature_col.extend(col[0:index])
        feature_data.extend(data[0:index])
        feature_row.extend([l[0] for i in range(feature_num)])

    sparse_feature = (feature_data, feature_row, feature_col, feature_indptr)
    sparse_label = (label_data, label_row, label_col, label_indptr)

    return sparse_feature, sparse_label

def NMI(feature,label,i,j,true_len):
    global feature_item
    add = feature_item
    j = j + add
    global H_feature, H_label, H_feature_lable
    if (i, j) in H_feature_lable:
        je = H_feature_lable[(i, j)]
    else:
        # diffDataCount_X is the horizontal component is distribution table
        diffDataCount_X = true_len

        # diffDataCount_Y is the vertical component is distribution table
        diffDataCount_Y = true_len

        distributionXY = np.zeros((diffDataCount_Y, diffDataCount_X))

        before_f = 0
        before_l = 0
        indptr_f = feature[3]
        indptr_l = label[3]
        diffDataX = []
        diffDataNumX = {}
        diffDataNumY = {}
        diffDataY = []
        for (index_f, index_l) in zip(indptr_f, indptr_l):

            data_f = feature[0][before_f:index_f]
            col_f = feature[2][before_f:index_f]
            before_f = index_f

            data_l = label[0][before_l:index_l]
            col_l = label[2][before_l:index_l]

            before_l = index_l
            i_exit = col_f.count(i)  # i是feature
            j_exit = col_l.count(j)  # j是label

            if (i_exit > 0):
                data_i = data_f[col_f.index(i)]
            else:
                data_i = 0

            if diffDataX.count(data_i) > 0:
                distribution_x = diffDataX.index(data_i)
                diffDataNumX[data_i] += 1
            else:
                diffDataX.append(data_i)
                distribution_x = len(diffDataX) - 1
                diffDataNumX[data_i] = 1

            if (j_exit > 0):
                data_j = data_l[col_l.index(j)]
            else:
                data_j = 0

            if diffDataY.count(data_j) > 0:
                distribution_y = diffDataY.index(data_j)
                diffDataNumY[data_j] += 1
            else:
                diffDataY.append(data_j)
                distribution_y = len(diffDataY) - 1
                diffDataNumY[data_j] = 1

            distributionXY[distribution_x][distribution_y] += 1

        diffDataCount_X = len(diffDataX)
        diffDataCount_Y = len(diffDataY)

        distributionXY = distributionXY[:diffDataCount_X, :diffDataCount_Y]
        distributionXY = distributionXY / true_len
        je = JointEntropy(distributionXY)
        H_feature_lable[(i, j)] = je
    if i in H_feature:
        HX = H_feature[i]
    else:
        HX = DataEntropy(true_len, diffDataNumX)
        H_feature[i] = HX
    if j in H_label:
        HY = H_label[j]
    else:
        HY = DataEntropy(true_len, diffDataNumY)
        H_label[j] = HY

    if (HX == 0.0 or HY == 0.0):
        return 0
    mi = HX + HY - je
    res = mi / math.sqrt(HX * HY)
    return res

def JointEntropy(distributionXY):
    je = 0
    [lenY, lenX] = np.shape(distributionXY)
    for i in range(lenY):
        for j in range(lenX):
            if (distributionXY[i][j] != 0):
                je = je - distributionXY[i][j] * math.log2(distributionXY[i][j])
    return je
def DataEntropy(dataArrayLen, diffDataNum):
    diffDataArrayLen = len(diffDataNum)
    entropyVal = 0;
    p=[]
    for i in diffDataNum:
        proptyVal = diffDataNum[i] / dataArrayLen
        p.append(proptyVal)
        if (proptyVal != 0):
            entropyVal = entropyVal - proptyVal * math.log2(proptyVal)
    return entropyVal

def distance(feature,i,j,true_len):
    global H_feature,H_featureS

    if (i,j) in H_featureS:
        je=H_featureS[(i,j)]
    else:
        # diffDataCount_X is the horizontal component is distribution table
        diffDataCount_X = true_len

        # diffDataCount_Y is the vertical component is distribution table
        diffDataCount_Y = true_len

        distributionXY = np.zeros((diffDataCount_Y, diffDataCount_X))

        before = 0
        indptr = feature[3]
        diffDataX = []
        diffDataY = []
        diffDataNumX = {}
        diffDataNumY = {}
        for index in indptr:
            data = feature[0][before:index]
            # row = feature[1][before:index]
            col = feature[2][before:index]
            before = index

            i_exit = col.count(i)
            j_exit = col.count(j)

            if (i_exit > 0):
                data_i = data[col.index(i)]
                # distribution_x=diffData1.index(data_i)
            else:
                data_i = 0
                # distribution_x=diffDataCount_X-1
            if diffDataX.count(data_i) > 0:
                distribution_x = diffDataX.index(data_i)
                diffDataNumX[data_i] += 1
            else:
                diffDataX.append(data_i)
                distribution_x = len(diffDataX) - 1
                diffDataNumX[data_i] = 1

            if (j_exit > 0):
                data_j = data[col.index(j)]
                # distribution_y=diffData2.index(data_j)
            else:
                data_j = 0
                # distribution_y=diffDataCount_Y-1
            if diffDataY.count(data_j) > 0:
                distribution_y = diffDataY.index(data_j)
                diffDataNumY[data_j] += 1
            else:
                diffDataY.append(data_j)
                distribution_y = len(diffDataY) - 1
                diffDataNumY[data_j] = 1

            distributionXY[distribution_x][distribution_y] += 1

        diffDataCount_X = len(diffDataX)
        diffDataCount_Y = len(diffDataY)

        distributionXY = distributionXY[:diffDataCount_X, :diffDataCount_Y]
        distributionXY = distributionXY / true_len
        je = JointEntropy(distributionXY)
        H_featureS[(i, j)] = je

    if i in H_feature:
        HX=H_feature[i]
    else:
        HX = DataEntropy(true_len, diffDataNumX)
        H_feature[i]=HX
    if j in H_feature:
        HY=H_feature[j]
    else:
        HY = DataEntropy(true_len, diffDataNumY)
        H_feature[j]=HY

    if je == 0.0:
        return 1
    return 2 - (HX + HY) / je

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

        self.NMI = readNMI(kwargs["NMI"],kwargs["n"],kwargs["l"])
        self.dis = readDis(kwargs["dis"],kwargs["n"])



    def mutation(self, s):
        rand_rate = 1.0 / (self.n)
        change = np.random.binomial(1, rand_rate, self.n)
        return np.abs(s - change)

    def doGSEMO(self, path):
        population = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
        self.tempOptimum = []
        fitness = np.mat(np.zeros([1, 2]))
        popSize = 1
        t = 0  # the current iterate count j
        sum = 0
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
            s = population[randint(1, popSize) - 1, :]  # choose a individual from population randomly
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
        values=[]
        for i in range(size):
            values.append(self.NMI[linklist[i][1]][l])
        values.sort(reverse=True)
        value=0
        for i in range(min(self.top,size)):
            value+=values[i]
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