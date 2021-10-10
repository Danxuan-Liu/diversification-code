import os
import random
import argparse
import numpy as np
import sys
import copy
import math

from os import listdir
from os.path import isfile, join
class Rel:
    def __init__(self, **kwargs):
        self.id_text = kwargs["id_text"]
        self.head_text = kwargs["head_text"]
        self.deprel = kwargs["deprel"]
class Node:
    def __init__(self, **kwargs):
        self.relations=[]
        for item in kwargs["relations"]:
            self.relations.append(item)
        self.document_id=kwargs["document_id"]
class Problem:
    def __init__(self, **kwargs):
        self.nodes = []
        self.weight = []
        self.distance = []
        self.similarity = []
        if "weight_text" in kwargs:
            ss = 0
            for line in kwargs["weight_text"].split("\n"):
                ss+=1
                if len(line) > 0:
                    rels = []
                    tokens = line.split(";")
                    info = tokens[0].split()
                    document_id = info[0]
                    weight = float(info[1])
                    self.weight.append(weight)
                    node = Node(document_id=document_id, relations=[])
                    self.nodes.append(node)
            self.k = kwargs["k"]
            self.n = len(self.nodes)
            self.mylambda = kwargs["mylambda"]
            self.alpha = kwargs["alpha"]
            self.beta = kwargs["beta"]
        if "dis_text" in kwargs:
            for line in kwargs["dis_text"].split("\n"):
                # print(line)
                d = []
                for value in line.split():
                    d.append(float(value))
                self.distance.append(d)
        if "similarity" in kwargs:
            for line in kwargs["similarity"].split("\n"):
                s=[]
                for value in line.split():
                    s.append(float(value))
                self.similarity.append(s)
    def evaluateObjective(self,temp,new_item):
        f_old=temp[1]
        f_new=0
        new=temp[0]+[new_item]

        f1 = 0
        f2 = 0
        f3 = 0
        div = 0
        for i in new:
            f1 += self.weight[i]

        dic = {}
        for i in new:
            name = self.nodes[i].document_id
            if name not in dic:
                dic[name] = 1
            else:
                dic[name] += 1
        for v in dic.values():
            f2 += math.sqrt(v)

        for u in new:
            a, b = 0, 0
            for v in range(len(self.similarity[u])):
                if v in new:
                    a += self.similarity[u][v]
                b += self.similarity[u][v]
            f3 += min(a, 0.25 * b)

        for i in range(len(temp[0])):
            div += self.distance[temp[0][i]][new_item]

        f_new=f1 + self.alpha * f2 + self.beta * f3
        res = 0.5*(f_new-f_old) + self.mylambda * div
        return res,f_new

    def select_best_item(self,res, items):
        evaluations=[]
        fs = []
        for s in items:
            res1 ,f=self.evaluateObjective(res,s)
            evaluations +=[res1]
            fs+=[f]
        index = np.argmax(evaluations)
        return (res[0] + [items.pop(index)],fs[index]),items

    def greedy(self):
        items=[]
        res=([],0)
        for i in range(self.n):
            items.append(i)
        while len(res[0]) < self.k:
            new_res,left_items = self.select_best_item(res, items)
            items=left_items
            res=new_res
        return res

    def exchange(self,solution,items):
        exits=False
        temp_solution = copy.deepcopy(solution)
        max_fitness=self.Calculate_true_value(temp_solution)
        max_solution = solution
        exchang_item=[0,0]
        for add_item in items:
            for del_item in solution[0]:
                temp_solution = copy.deepcopy(solution)
                temp_solution[0].pop(temp_solution[0].index(del_item))
                temp_solution[0].append(add_item)
                new_solution =temp_solution
                new_fitness=self.Calculate_true_value(new_solution)
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
        for i in init_res[0]:
            items.pop(items.index(i))


        while True:
            flag,value,res,exchang_item=self.exchange(res,items)
            if flag==False:
                break
            else:
                items.pop(items.index(exchang_item[0]))
                items.append(exchang_item[1])

        return res,value

    def Calculate_true_value(self,res):
        size = len(res[0])
        # f=res[1]
        f1 = 0
        f2 = 0
        f3 = 0
        div = 0
        for i in res[0]:
            f1 += self.weight[i]

        dic = {}
        for i in res[0]:
            name = self.nodes[i].document_id
            if name not in dic:
                dic[name] = 1
            else:
                dic[name] += 1
        for v in dic.values():
            f2 += math.sqrt(v)

        for u in res[0]:
            a, b = 0, 0
            for v in range(len(self.similarity[u])):
                if v in res[0]:
                    a += self.similarity[u][v]
                b += self.similarity[u][v]
            f3 += min(a, 0.25 * b)

        f = f1 + self.alpha * f2 + self.beta * f3

        for i in range(size):
            for j in range(i + 1, size):
                div += self.distance[res[0][i]][res[0][j]]
        v = f + self.mylambda * div

        return v


def OutputResult(res,value,log):
    log.write(str(value))
    log.write("\n")
    for item in res:
        log.write(str(item+1))
        log.write(' ')
    log.write("\n")
    log.close()

def OutputAvg(all_greedy,all_ls,log):
    log.write("Greedy: ")
    log.write(str(np.mean(all_greedy)))
    log.write("\n")
    log.write("LocalSearch: ")
    log.write(str(np.mean(all_ls)))
    log.write("\n")
    log.close()

def run_experiments(args):
    new_address=args.folder
    folders = [f for f in listdir(new_address)]
    num = 0
    all_greedy=[]
    all_ls=[]
    for folder in folders:
        print("run " + str(args.constraint) + ":" + str(folder))
        address = new_address + "/" + folder

        f1 = open(address + "/" + folder + "_weight.txt",encoding='ISO-8859-1')
        weight_text = f1.read()

        f2 = open(address + "/" + folder + "_distance.txt")
        dis_text = f2.read()

        f3 = open(address + "/" + folder + "_similarity.txt")
        similarity = f3.read()

        problem = Problem(dis_text=dis_text, weight_text=weight_text, similarity=similarity,k=args.constraint, mylambda=args.mylambda,
                          alpha=args.alpha, beta=args.beta)

        save_address = args.save_file + "/" + str(args.constraint)
        if not os.path.exists(save_address):
            os.makedirs(save_address)

        print("doing greedy...")
        res=problem.greedy()
        value = problem.Calculate_true_value(res)
        all_greedy.append(value)

        print("saving greedy...")
        save_address = args.save_file + "/" + str(args.constraint)
        if not os.path.exists(save_address):
            os.makedirs(save_address)
        log = open(save_address + "/greedy_" + folder + ".txt", "w")
        OutputResult(res[0], value, log)

        print("doing LS...")
        res,value=problem.LocalSearch(res)
        all_ls.append(value)

        print("saving LS...")
        log = open(save_address + "/LS_" + folder + ".txt", "w")
        OutputResult(res[0], value, log)

        num+=1

    log = open(save_address+ "/Average.txt" , "w") if args.save_file else sys.stdout
    OutputAvg(all_greedy,all_ls,log)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-folder', default='data')
    argparser.add_argument('-save_file', help="save_result", default='Result_GreedyLS', type=str)
    argparser.add_argument('-n_instances', help="number of instances", default=50, type=int)
    argparser.add_argument('-constraint', help="max size of subset", default=20, type=int)
    argparser.add_argument('-mylambda', help="trade_off", type=float, default=1)
    argparser.add_argument('-alpha', type=float, default=0.2)
    argparser.add_argument('-beta', type=float, default=0.2)

    args = argparser.parse_args()
    args.save_file = f'Result_GreedyLS/{args.mylambda}'
    run_experiments(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
