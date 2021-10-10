import os
import argparse
import GSEMO as gsm
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
        self.nodes=[]
        self.weight = []
        self.distance = []
        self.similarity=[]
        if "weight_text" in kwargs:
            ss=0
            for line in kwargs["weight_text"].split("\n"):
                if len(line)>0:
                    # print("------------------")
                    # print(line)
                    rels=[]
                    tokens=line.split(";")
                    # print(tokens)
                    info=tokens[0].split()
                    document_id=info[0]
                    weight=float(info[1])
                    self.weight.append(weight)

                    # id_text=info[2]
                    # head_text=info[3]
                    # deprel=info[4]
                    # rel=Rel(id_text=id_text,head_text=head_text,deprel=deprel)
                    # rels.append(rel)
                    # for i in range(1,len(tokens)-1):
                    #     info=tokens[i].split()
                    #     if len(info)>3:
                    #         id_text = info[0]
                    #         head_text = info[1]
                    #         deprel = info[2]
                    #         rel = Rel(id_text=id_text, head_text=head_text, deprel=deprel)
                    #         rels.append(rel)
                    node=Node(document_id=document_id,relations=[])
                    self.nodes.append(node)
            self.k = kwargs["k"]
            self.n = len(self.nodes)
            self.mylambda = kwargs["mylambda"]
            self.alpha=kwargs["alpha"]
            self.beta=kwargs["beta"]
        if "dis_text" in kwargs:
            for line in kwargs["dis_text"].split("\n"):
                # print(line)
                d=[]
                for value in line.split():
                    d.append(float(value))
                self.distance.append(d)
        if "similarity" in kwargs:
            for line in kwargs["similarity"].split("\n"):
                s=[]
                for value in line.split():
                    s.append(float(value))
                self.similarity.append(s)
    def Gsemo(self,path):
        gesmo=gsm.GSEMO(k=self.k,n=self.n,w=self.weight,dis=self.distance,similarity=self.similarity,mylambda=self.mylambda,alpha=self.alpha,beta=self.beta,nodes=self.nodes)
        iterationoTime=math.exp(1)*self.n*self.k*self.k*self.k/2
        gesmo.setIterationTime(iterationoTime)
        return gesmo.doGSEMO(path)



def run_experiments(args):
    new_address=args.folder
    folders = [f for f in listdir(new_address) ]
    for folder in folders:
        print("run " + str(args.constraint) + ":" + str(folder))
        address = new_address + "/" + folder
        #files= [f for f in listdir(address) if isfile(join(address, f))]

        f1=open(address + "/" + folder+"_weight.txt",encoding='ISO-8859-1')
        weight_text = f1.read()

        f2 = open(address + "/" + folder + "_distance.txt")
        dis_text = f2.read()

        f3 = open(address + "/" + folder + "_similarity.txt")
        similarity = f3.read()

        problem = Problem(dis_text=dis_text,weight_text=weight_text,similarity=similarity, k=args.constraint, mylambda=args.mylambda,alpha=args.alpha,beta=args.beta)

        save_address = args.save_file + "/" + str(args.constraint)
        if not os.path.exists(save_address):
            os.makedirs(save_address)

        path=save_address+"/Result_"+ folder+".txt"

        result=problem.Gsemo(path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-folder', default='data')
    argparser.add_argument('-save_file', help="save_result", default='Result_Gsemo',type=str)
    argparser.add_argument('-n_instances', help="number of instances", default=50, type=int)
    argparser.add_argument( '-constraint', help="max size of subset", default=15, type=int)
    argparser.add_argument('-mylambda', help="trade_off", type=float, default=1)
    argparser.add_argument('-alpha', type=float, default=0.2)
    argparser.add_argument('-beta', type=float, default=0.2)

    args = argparser.parse_args()
    args.save_file = f'Result_Gsemo/{args.mylambda}'
    run_experiments(args)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
