import stanza
import argparse
import re
import os
import random
import sys
import numpy as np

import time, datetime
from os import listdir
from os.path import isfile, join
from nltk.corpus import wordnet as wn

global popularity, edgeWeight, similarity
popularity = {}
edgeWeight = []
similarity = []


class Rel:
    def __init__(self, **kwargs):
        self.id = kwargs["id"]
        self.id_text = kwargs["id_text"]
        self.head = kwargs["head"]
        self.head_text = kwargs["head_text"]
        self.deprel = kwargs["deprel"]


class Node:
    def __init__(self, **kwargs):
        self.relations = []
        for item in kwargs["relations"]:
            self.relations.append(item)
        self.document_id = kwargs["document_id"]

    def judgeRel(self, rel):
        for r in self.relations:
            if r == rel:
                return True
        return False


def num_of_document(node, C, list):
    res = 0
    for c in list:
        if c != C:
            flag = False
            for n in c:
                for rel in n.relations:
                    if node.judgeRel(rel):
                        res += 1
                        flag = True
                        break
                if flag == True:
                    break
    return res


def compute_popularity(list):
    global popularity
    sum = 0
    for c in list:
        for node in c:
            if node not in popularity:
                popularity[node] = num_of_document(node, c, list)
                sum += popularity[node]
    # normalize
    if sum != 0:
        for k in popularity.keys():
            popularity[k] /= sum


def compute_edgeWeight(nodes):
    global edgeWeight, similarity
    edgeWeight = []
    similarity=[]

    for i in range(len(nodes)):
        row_list = []
        max = 0
        for j in range(i + 1, len(nodes)):
            sim=random.random()
            row_list.append(sim)
            if sim>max:
                max=sim
        similarity.append(row_list)

    for list in similarity:
        row_list = []
        for v in list:
            v = 1 - v
            row_list.append(v)
        edgeWeight.append(row_list)



def compute_similarity(node1, node2):
    sim = 0
    for rel1 in node1.relations:
        for rel2 in node2.relations:
            if rel1.deprel == rel2.deprel:
                sim += WN(rel1.id_text, rel2.id_text) * WN(rel1.head_text, rel2.head_text)
    return sim


def WN(w1, w2):
    a = wn.synsets(w1)
    b = wn.synsets(w2)
    if len(a) > 0 and len(b) > 0:
        for i in a:
            for j in b:
                sim = i.path_similarity(j)
                return sim
    else:
        return 0


def compute_distance(sentences_num, log):
    NUMBER = sentences_num
    Graph_Matrix = [[0] * NUMBER for row in range(NUMBER)]
    distance = [[0] * NUMBER for row in range(NUMBER)]

    for i in range(NUMBER):
        for j in range(NUMBER):
            if i == j:
                Graph_Matrix[i][j] = 0
            elif i < j:
                Graph_Matrix[i][j] = edgeWeight[i][j - i - 1]
            else:
                Graph_Matrix[i][j] = Graph_Matrix[j][i]

    for i in range(NUMBER):
        for j in range(NUMBER):
            distance[i][j] = Graph_Matrix[i][j]
            distance[j][i] = Graph_Matrix[i][j]

    for k in range(NUMBER):
        for i in range(NUMBER):
            for j in range(NUMBER):
                if distance[i][k] + distance[k][j] < distance[i][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
    for i in range(NUMBER):
        for j in range(NUMBER):
            log.write(str(distance[i][j]))
            log.write(" ")
        log.write("\n")
    log.close()


def save_weight(log):
    for k in popularity.keys():
        log.write(k.document_id)
        log.write(" ")
        log.write(str(popularity[k]))
        log.write(" ")
        for rel in k.relations:
            log.write(rel.id_text)
            log.write(" ")
            log.write(rel.head_text)
            log.write(" ")
            log.write(rel.deprel)
            log.write(";")
        log.write("\n")
    log.close()

def save_sim(sentences_num, log):
    NUMBER = sentences_num
    sim = [[0] * NUMBER for row in range(NUMBER)]
    for i in range(NUMBER):
        for j in range(NUMBER):
            if i == j:
                sim[i][j] = 1
            elif i < j:
                sim[i][j] = similarity[i][j - i - 1]
            else:
                sim[i][j] = sim[j][i]

    for i in range(NUMBER):
        for j in range(NUMBER):
            log.write(str(sim[i][j]))
            log.write(" ")
        log.write("\n")
    log.close()

def operateData(args):
    new_address = args.folder
    folders = [f for f in listdir(new_address)]
    s = re.compile(
        '(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=\.|\?|\!)\"*\s*\s*(?:\W*)([A-Z])')
    stanza.download('en')
    en_nlp = stanza.Pipeline('en')
    for folder in folders:
        global popularity
        popularity = {}
        address = new_address + "/" + folder
        files = [f for f in listdir(address) if isfile(join(address, f))]
        all_rel_list = []
        for file in files:
            rel_list = []
            f = open(address + "/" + file, "r")
            instance = f.read()
            a = s.finditer(instance)
            pre = 0
            for i in a:
                sen = instance[pre:i.span()[0]]
                pre = i.span()[0] + 1
                doc = en_nlp(sen)
                relations = []
                for sent in doc.sentences:
                    for word in sent.words:
                        rel = Rel(id=word.id, id_text=word.text, head=word.head,
                                  head_text=sent.words[word.head - 1].text if word.head > 0 else "root",
                                  deprel=word.deprel)
                        relations.append(rel)
                node = Node(relations=relations, document_id=file)
                rel_list.append(node)
            all_rel_list.append(rel_list)

        compute_popularity(all_rel_list)

        save_address = args.save_file + "/" + folder
        if not os.path.exists(save_address):
            os.makedirs(save_address)
        log = open(save_address+ "/" + folder + "_weight.txt", "w")
        save_weight(log)

        nodes = []
        for k in popularity.keys():
            nodes.append(k)

        compute_edgeWeight(nodes)

        log = open(save_address  + "/" + folder + "_distance.txt", "w")
        compute_distance(len(nodes), log)

        log = open(save_address + "/" + folder + "_similarity.txt", "w")
        save_sim(len(nodes), log)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-folder', default='DUC_2004', type=str)
    argparser.add_argument('-save_file', help="save_result", default='data', type=str)
    args = argparser.parse_args()

    operateData(args)