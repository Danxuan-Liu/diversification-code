import os
import random
from random import randint
import argparse
import sys
import GSEMO as prd
import numpy as np
import math
import copy


class Problem:
    def __init__(self, **kwargs):
        self.weight = []
        self.distance = []
        if "text" in kwargs:
            it=-1
            for line in kwargs["text"].split("\n"):
                if line !="":
                    if it==-1:
                        for w in line.split():
                            self.weight.append(float(w))
                    else:
                        b=[]
                        for value in line.split():
                            b.append(float(value))
                        self.distance.append(b)
                    it+=1
        self.k=kwargs["k"]
        self.n=kwargs["n"]
        self.mylambda=kwargs["mylambda"]
        self.iteration = kwargs["iteration"]
        self.changes_per_time=kwargs["changes_per_time"]

        self.pord = prd.GSEMO(k=self.k, n=self.n, w=self.weight, dis=self.distance, mylambda=self.mylambda,
                               iteration=self.iteration)
        iterationoTime = math.exp(1) * self.n * self.k * self.k * self.k / 2
        self.pord.setIterationTime(iterationoTime)

    def GSEMO(self):
        return self.pord.GSEMO()

    def Calucalate_true_value_pord(self,res):
        index = np.nonzero(res)
        size = np.shape(index)[1]

        linklist = []
        for i, j in zip(index[0], index[1]):
            linklist.append([i, j])

        f = 0
        div = 0
        for i in linklist:
            f += self.weight[i[1]]
        for i in range(size):
            for j in range(i + 1, size):
                div += self.distance[linklist[i][1]][linklist[j][1]]
        res = f + self.mylambda * div
        return res

    def Calculate_true_value_ls(self,res):
        size = len(res)
        f = 0
        div = 0
        for i in res:
            f += self.weight[i]
        for i in range(size):
            for j in range(i + 1, size):
                div += self.distance[res[i]][res[j]]
        res = f + self.mylambda * div
        return res

    def evaluateObjective(self,temp,new_item):
        size=len(temp)
        f=0
        div=0
        f=self.weight[new_item]
        for i in range(size):
            div+=self.distance[temp[i]][new_item]
        res=0.5*f+self.mylambda*div
        return res

    def select_best_item(self,res, items):
        evaluations = [self.evaluateObjective(res,s) for s in items]
        index = np.argmax(evaluations)
        return res + [items.pop(index)],items

    def greedy(self):
        items=[]
        res=[]
        value=0
        for i in range(self.n):
            items.append(i)
        while len(res) < self.k:
            new_res,left_items = self.select_best_item(res, items)
            items=left_items
            res=new_res
        return res

    def exchange(self,solution,items):
        exits=False
        temp_solution = copy.deepcopy(solution)
        max_fitness=self.Calculate_true_value_ls(temp_solution)
        max_solution = solution
        exchang_item=[0,0]
        for add_item in items:
            for del_item in solution:
                temp_solution = copy.deepcopy(solution)
                temp_solution.pop(temp_solution.index(del_item))
                temp_solution.append(add_item)
                new_solution =temp_solution
                new_fitness=self.Calculate_true_value_ls(new_solution)
                if(new_fitness>max_fitness):
                    exits=True
                    max_fitness=new_fitness
                    max_solution=new_solution
                    exchang_item[0]=add_item
                    exchang_item[1]=del_item

        return exits,max_fitness,max_solution,exchang_item

    def LocalSearch(self,init_res):
        items = []
        res=init_res
        value = 0
        for i in range(self.n):
            items.append(i)
        for i in init_res:
            items.pop(items.index(i))

        #start with greedy algorithm solution
        sum_iteration = 0
        while sum_iteration<self.iteration:
            flag,value,res,exchang_item=self.exchange(res,items)
            if flag==True:
                items.pop(items.index(exchang_item[0]))
                items.append(exchang_item[1])
            sum_iteration+=self.k*(self.n-self.k)

        return res,value

    def change(self,change_type):
        if change_type=="MPERTURBATION":
            for step in range(self.changes_per_time):
                self.MPERTURBATION()
        elif change_type=="VPERTURBATION":
            for step in range(self.changes_per_time):
                self.VPERTURBATION()
        elif change_type=="EPERTURBATION":
            for step in range(self.changes_per_time):
                self.EPERTURBATION()
    def MPERTURBATION(self):
        p=random.random()
        if p<0.5:
            self.VPERTURBATION()
        else:
            self.EPERTURBATION()
    def VPERTURBATION(self):
        index=randint(1, self.n) - 1
        w=random.random()
        self.weight[index]=w
        self.pord.change_weight(index,w)
    def EPERTURBATION(self):
        u=randint(1, self.n) - 1
        while True:
            v=randint(1, self.n) - 1
            if u!=v:
                break
        w=random.random()+1
        self.distance[u][v]=w
        self.distance[v][u]=w
        self.pord.change_distance(u,v,w)

def Output_Avg_Ratio(matrix_ls,matrix_pord,log):
    log.write("Local Search: ")
    ls_mean=matrix_ls.mean(axis=0)
    log.write(str(ls_mean))
    log.write("\n")
    log.write("GSEMO: ")
    gsemo_mean = matrix_pord.mean(axis=0)
    log.write(str(gsemo_mean))
    log.write("\n")
    log.write("---------------------------------")
    log.write('\n')
    log.write(str(ls_mean/gsemo_mean))
    log.write('\n')
    log.close()


def output_SigleResult(pord,ls,log):
    log.write("LocalSearch: ")
    log.write(str(ls))
    log.write(" GSEMO: ")
    log.write(str(pord))
    log.write("\n")
    log.write("------------------------------------------------")
    log.write("\n")
    log.close()

def generate_data(n_instances, address,n_items):
    txt_id=0
    for i in range(n_instances):
        print("generate " + str(i))
        txt_id=txt_id+1

        directory = address + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        name = "instance_" + str(txt_id) + ".txt"
        f = open(directory + name, "w")

        text = ""
        for n in range(n_items):
            weight=random.random()
            text=text+str(weight)+" "
        text = text + '\n'
        f.write(text)
        distance=[[0 for col in range(n_items)] for row in range(n_items)]
        for u in range(n_items):
            for v in range(n_items):
                if u==v:
                    distance[u][v]=0
                elif v<u:
                    distance[u][v] = distance[v][u]
                else:
                    distance[u][v]=1+random.random()
                f.write(str(distance[u][v]) + " ")
            f.write('\n')
        f.close()

def run_experiments(args):

    print("run " + str(args.constraint)+":"+args.file)
    args.iteration=int(args.iteration)

    #load file
    f = open(args.folder + "/" + args.file, "r")
    instance = f.read()
    save_address = args.save_file
    if not os.path.exists(save_address):
        os.makedirs(save_address)

    #initial Problem
    problem=Problem(text=instance,k=args.constraint,n=args.n_items,mylambda=args.mylambda,iteration=args.iteration,changes_per_time=args.changes_per_time)
    #do greedy solution
    init_solution = problem.greedy()

    #format convert
    zero_solution = np.zeros([1, args.n_items], 'int8')
    for i in init_solution:
        zero_solution[0][i] = 1
    res_gsemo = np.mat(zero_solution)
    #set pord's initial solution
    problem.pord.setInilSolution(res_gsemo)
    #set ls's initial solution
    res_ls=init_solution

    for change_time in range(args.change_times):
        problem.change(args.change_type)

        res_pord=problem.GSEMO()
        vaule_pord=problem.Calucalate_true_value_gsemo(res_gsemo)

        res_ls, value_ls = problem.LocalSearch(res_ls)
        log = open(save_address + "/Result_" + args.file, 'a')
        output_SigleResult(vaule_pord,value_ls,log)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-folder', default='SyntheticData/500')
    argparser.add_argument('-file', default='instance_1.txt')
    argparser.add_argument('-p', '--save_file', help="save_result", default='Result/500',type=str)
    argparser.add_argument('-s', '--n_items', help="number of items", default=500, type=int)
    argparser.add_argument('-constraint', help="max size of subset", default=20, type=int)
    argparser.add_argument('-iteration', default=1, type=int)
    argparser.add_argument('-mylambda', help="trade_off", type=float, default=0.2)
    argparser.add_argument('-change_type',default="MPERTURBATION",type=str)
    argparser.add_argument('-change_times', type=int, default=50)
    argparser.add_argument('-changes_per_time', type=int, default=50)
    argparser.add_argument('-mul', type=int, default=10)
    args = argparser.parse_args()
    args.save_file = f'Result/{str(args.constraint)+"/"+str(args.mul)+"kn_"+str(args.changes_per_time)+"_"+str(args.mylambda)+"_"+str(args.change_times)}'

    args.iteration=f'{args.constraint * args.n_items * args.mul}'
    run_experiments(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
