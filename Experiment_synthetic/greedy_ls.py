import os
import random
import argparse
import numpy as np
import sys
import copy

from os import listdir
from os.path import isfile, join
class Problem:
    def __init__(self, **kwargs):
        self.weight=[]
        self.distance=[]
        if "text" in kwargs:
            it = -1
            for line in kwargs["text"].split("\n"):
                if line != "":
                    if it == -1:
                        for w in line.split():
                            self.weight.append(float(w))
                    else:
                        b = []
                        for value in line.split():
                            b.append(float(value))
                        self.distance.append(b)
                    it += 1
        self.k = kwargs["k"]
        self.n = kwargs["n"]
        self.mylambda = kwargs["mylambda"]


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
        max_fitness=self.Calculate_true_value(temp_solution)
        max_solution = solution
        exchang_item=[0,0]
        for add_item in items:
            for del_item in solution:
                temp_solution = copy.deepcopy(solution)
                temp_solution.pop(temp_solution.index(del_item))
                temp_solution.append(add_item)
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
        for i in range(self.n):
            items.append(i)
        for i in init_res:
            items.pop(items.index(i))
        #start with greedy solution
        while True:
            flag,value,res,exchang_item=self.exchange(res,items)
            if flag==False:
                break
            else:
                items.pop(items.index(exchang_item[0]))# delete exchange_item[0] from V\X
                items.append(exchang_item[1])#add exchang_item[1] in V\X
        return res,value

    def Calculate_true_value(self,res):
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

def generate_data(n_instances, address, cadinality,n_items):
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
    if args.new.lower() == "true":
        address = args.folder +"/"
        if not os.path.exists(address):
            os.makedirs(address)
        print("created\n\t" + address)
        generate_data(args.n_instances, address, args.cadinality,args.n_items)
        print("file saved in \n\t" + address)
    else:
        new_address=args.folder
        files = [f for f in listdir(new_address) if isfile(join(new_address, f))]
        num = 0
        all_greedy=[]
        all_ls=[]
        for file in files:
            print("run " + str(args.constraint)+":"+str(file))
            f = open(new_address + "/" + file, "r")
            instance  = f.read()
            save_address=args.save_file+"/"+str(args.constraint)
            if not os.path.exists(save_address):
                os.makedirs(save_address)
            problem = Problem(text=instance, k=args.constraint, n=args.n_items, mylambda=args.mylambda)
            # doing greedy
            print("doing greedy...")
            res=problem.greedy()
            value = problem.Calculate_true_value(res)
            all_greedy.append(value)
            # saving greedy
            print("saving greedy...")
            save_address = args.save_file + "/" + str(args.constraint)
            if not os.path.exists(save_address):
                os.makedirs(save_address)

            log = open(save_address + "/greedy_" + file + ".txt", "w")
            OutputResult(res, value, log)
            # doing LS
            print("doing LS...")
            res,value=problem.LocalSearch(res)
            all_ls.append(value)
            # saving LS
            print("saving LS...")
            log = open(save_address + "/LS_" + file + ".txt", "w")
            OutputResult(res, value, log)

            num+=1

        save_address = args.save_file+"/"+str(args.constraint)
        if not os.path.exists(save_address):
            os.makedirs(save_address)
        log = open(save_address+ "/Average.txt" , "w") if args.save_file else sys.stdout
        OutputAvg(all_greedy,all_ls,log)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-f', '--folder', default='SyntheticData/500')

    argparser.add_argument('-n', '--new', help="create new dataset", default="false")
    argparser.add_argument('-p', '--save_file', help="save_result", default='Result_GreedyLS/500',type=str)

    argparser.add_argument('-n_instances', help="number of instances", default=50, type=int)
    argparser.add_argument('-n_items', help="number of items", default=500, type=int)
    argparser.add_argument('-constraint', help="max size of subset", default=5, type=int)
    argparser.add_argument('-mylambda', help="trade_off", type=float, default=1)

    args = argparser.parse_args()
    args.save_file = f'Result_GreedyLS/{args.mylambda}'
    args.folder = f'SyntheticData/{args.n_items}'
    run_experiments(args)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
