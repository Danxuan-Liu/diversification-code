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
        self.dic = {}
        self.distance = []
        self.pair={}
        if "text" in kwargs:
            for line in kwargs["text"].split("\n"):
                tokens=line.split( )
                if len(tokens)>=2:
                    if tokens[1] in self.dic:
                        self.dic[tokens[1]].append(line)
                    else:
                        self.dic[tokens[1]] = [line]
        for d in self.dic:
            self.pair[d]=len(self.dic[d])

def output(data,address):
    for d in data.dic:
        index=d.split(":")
        save_address = address + "/"
        if not os.path.exists(save_address):
            os.makedirs(save_address)
        name=index[0]+index[1]+".txt"
        log = open(save_address+name, "w")
        print("write:"+save_address+name)
        for line in data.dic[d]:
            log.write(line)
            log.write("\n")
        log.close()

    save_address = address + "/"
    if not os.path.exists(save_address):
        os.makedirs(save_address)
    save_address = save_address + "info.txt"
    print("write:" + save_address )
    log = open(save_address, "w")
    for d in data.pair:
        log.write(d+" "+str(data.pair[d])+"\n")
    log.close()

    save_address = address + "/"
    if not os.path.exists(save_address):
        os.makedirs(save_address)
    save_address = save_address + "info_sort.txt"
    print("write:" + save_address)
    log = open(save_address, "w")
    pair_sort=sorted(data.pair.items(), key=lambda kv: (kv[1], kv[0]))
    for d in pair_sort:
        print("----------------------")
        print(d)
        log.write(d[0] + " " + str(d[1]) + "\n")
    log.close()



def readData(str):
    f = open(str, "r")
    instance = text = f.read()
    data=Dataset(text=instance)
    f.close()
    return data
def operate(args):
    data=readData(args.read)
    output(data,args.save)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-read', default='Data/original_data.txt',type=str)
    argparser.add_argument('-save', help="save_result", default='Data',type=str)
    args = argparser.parse_args()
    operate(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
