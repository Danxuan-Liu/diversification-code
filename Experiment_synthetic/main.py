import os
import random
import argparse
import GSEMO as gsm
import math
from os import listdir
from os.path import isfile, join
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

    def Gsemo(self,path):
        gesmo=gsm.GSEMO(k=self.k,n=self.n,w=self.weight,dis=self.distance,mylambda=self.mylambda)
        iterationoTime=math.exp(1)*self.n*self.k*self.k*self.k/2
        gesmo.setIterationTime(iterationoTime)
        return gesmo.doGSEMO(path)

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
    if args.new.lower() == "true":
        address = args.folder +"/"
        if not os.path.exists(address):
            os.makedirs(address)
        print("created\n\t" + address)
        generate_data(args.n_instances, address,args.n_items)
        print("file saved in \n\t" + address)
    else:
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
    argparser.add_argument('-f', '--folder', default='SyntheticData/500')
    argparser.add_argument('-n', '--new', help="create new dataset", default="false")
    argparser.add_argument('-p', '--save_file', help="save_result", default='Result_Gsemo/500',type=str)
    argparser.add_argument('-n_instances', help="number of instances", default=50, type=int)
    argparser.add_argument( '-n_items', help="number of items", default=500, type=int)
    argparser.add_argument( '-constraint', help="max size of subset", default=5, type=int)
    argparser.add_argument('-mylambda', help="trade_off", type=float, default=0.8)
    args = argparser.parse_args()
    args.save_file = f'Result_Gsemo/{args.mylambda}'
    args.folder = f'SyntheticData/{args.n_items}'
    run_experiments(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
