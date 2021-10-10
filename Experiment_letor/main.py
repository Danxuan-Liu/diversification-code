import os
import random
import argparse
import GSEMO as gsm
import numpy as np
import math
from os import listdir
from os.path import isfile, join
class Problem:
    def __init__(self, **kwargs):
        self.weight = []
        self.vector = []
        if "text" in kwargs:
            it = -1
            for line in kwargs["text"].split("\n"):
                tokens = line.split()
                size = len(tokens)
                if size >= 3:
                    self.weight.append(float(tokens[0]))
                    self.name = tokens[1]
                    change = tokens[2:]
                    f_vector = [float(i) for i in change]
                    self.vector.append(f_vector)
        self.k = kwargs["k"]
        self.n = kwargs["n"]
        self.mylambda = kwargs["mylambda"]

        self.distance = [[0 for col in range(self.n)] for row in range(self.n)]
        for u in range(len(self.vector)):
            for v in range(len(self.vector)):
                if u == v:
                    self.distance[u][v] = 0
                elif v < u:
                    self.distance[u][v] = self.distance[v][u]
                else:
                    self.distance[u][v] = self.cosine_similarity(self.vector[u], self.vector[v])

    def Gsemo(self,path):
        gesmo=gsm.GSEMO(k=self.k,n=self.n,w=self.weight,dis=self.distance,mylambda=self.mylambda)
        iterationoTime=math.exp(1)*self.n*self.k*self.k*self.k/2
        gesmo.setIterationTime(iterationoTime)
        return gesmo.doGSEMO(path)

    def bit_product_sum(self,x, y):
        return sum([item[0] * item[1] for item in zip(x, y)])

    def cosine_similarity(self,x, y, norm=False):
        # compute cosine similarity of vector x and vector y
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return 0.5 * cos + 0.5 if norm else cos  # Normalized to the interval [0, 1]

def run_experiments(args):
    new_address=args.folder
    files = [f for f in listdir(new_address) if isfile(join(new_address, f))]
    for file in files:
        print("run " + str(args.constraint)+":"+str(file))
        f = open(new_address + "/" + file, "r")
        instance = text = f.read()
        save_address = args.save_file + "/" + str(args.constraint)
        if not os.path.exists(save_address):
            os.makedirs(save_address)

        path=save_address+"/Result_" + file
        problem=Problem(text=instance,k=args.constraint,n=args.n_items,mylambda=args.mylambda)
        result=problem.Gsemo(path)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-f', '--folder', default='Experiment_Data/370')
    argparser.add_argument('-n', '--new', help="create new dataset", default="false")
    argparser.add_argument('-p', '--save_file', help="save_result", default='Result_Gsemo/370',type=str)
    argparser.add_argument('-n_instances', help="number of instances", default=50, type=int)
    argparser.add_argument( '-n_items', help="number of items", default=370, type=int)
    argparser.add_argument( '-constraint', help="max size of subset", default=5, type=int)
    argparser.add_argument('-mylambda', help="trade_off", type=float, default=1)
    args = argparser.parse_args()
    args.save_file = f'Result_Gsemo/{args.mylambda}'
    args.folder = f'Experiment_Data/{args.n_items}'
    run_experiments(args)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
