# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime
import os
import random
import argparse
import sys
import numpy as np
import math
from os import listdir
from os.path import isfile, join
class Dataset:
    def __init__(self, **kwargs):
        self.documents=[]
        if "text" in kwargs:
            for line in kwargs["text"].split("\n"):
                tokens=line.split( )
                size=len(tokens)
                if size>=3:
                    lable=tokens[0]
                    index=tokens[1].split(":")
                    self.name=index[0]+index[1]
                    vector=[]
                    for i in range(2,size):
                        index=tokens[i].split(":")
                        vector.append(index[1])
                    tup=(lable,vector)
                    self.documents.append(tup)
        self.top=kwargs["items"]
        self.documents.sort(key=self.takeLable,reverse=True)
        for i in range(self.top,len(self.documents)):
            self.documents.pop(self.top)
    def takeLable(self,elem):
        return elem[0]

def output(data,address):
    save_address = address + "/"
    if not os.path.exists(save_address):
        os.makedirs(save_address)
    name = "instance_" + data.name + ".txt"
    log = open(save_address + name, "w")
    print("write:" + save_address + name)
    for d in data.documents:
        log.write(d[0]+" ")
        log.write(data.name+" ")
        for item in d[1]:
            log.write(item+" ")
        log.write("\n")
    log.close()




def generate(args):
    str=args.read
    f = open(str, "r")
    instance = f.read()
    f.close()
    for line in instance.split("\n"):
        pair=line.split(" ")
        if len(pair)==2:
            qid = pair[0]
            num = pair[1]
            read_address = args.file + "/"
            if not os.path.exists(read_address):
                os.makedirs(read_address)
            index = qid.split(":")
            name = index[0] + index[1]
            read_address = read_address + name + ".txt"
            f = open(read_address, "r")
            instance = f.read()
            data = Dataset(text=instance, items=args.top)
            f.close()
            output(data, args.save)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-read', default='Data/info_sort.txt',type=str)
    argparser.add_argument('-file', default='Data', type=str)
    argparser.add_argument('-top', default=370, type=int)

    argparser.add_argument('-save', help="save_result", default='Experiment_Data/370',type=str)
    args = argparser.parse_args()
    args.save = f'Experiment_Data/{args.top}'
    generate(args)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
