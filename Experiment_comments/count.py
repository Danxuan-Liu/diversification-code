import csv
import pandas as pd
from os import listdir
from os.path import isfile, join

birth_data = []
address="Comments"
dic={}
text={}
files= [f for f in listdir(address) if isfile(join(address, f))]
for file in files:
    f = csv.reader(open(address+"/"+file, 'r',encoding='ISO-8859-1'))
    num=0
    for i in f:
        if num!=0:
            text[i[1]]=i[0]
            if i[1] in dic:
                dic[i[1]]+=1
            else:
                dic[i[1]]=1
        num += 1
    log = open(address+"/"+"50_id.txt", "w")
    for v in dic:
        if dic[v] <= 200 and dic[v] > 100:
            print(dic[v])
            log.write(v)
            log.write(" ")
            log.write(str(dic[v]))
            log.write("\n")


    log.close()

