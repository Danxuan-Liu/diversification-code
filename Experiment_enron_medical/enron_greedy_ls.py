import os
import argparse
import numpy as np
import copy
def readNMI(address,row,col):
    f = open(address)
    nmi = np.zeros((row, col))

    text=f.read()
    lines=text.split("\n")
    i,j=0,0
    for line in lines:
        tokens=line.split(" ")
        for v in range(len(tokens)-1):
            nmi[i][j]=float(tokens[v])
            j+=1
        i+=1
        j=0
    return nmi
def readDis(address,n):
    f = open(address)
    dis = np.zeros((n, n))

    text = f.read()
    lines = text.split("\n")
    i, j = 0, 0
    for line in lines:
        j=i+1
        tokens = line.split(" ")
        for v in range(len(tokens) - 1):
            dis[i][j] = float(tokens[v])
            j += 1
        i += 1
    for i in range(1,n):
        for j in range(i):
            dis[i][j]=dis[j][i]
    return dis
class Problem:
    def __init__(self, **kwargs):
        self.k = kwargs["k"]
        self.n = kwargs["n"]
        self.mylambda = kwargs["mylambda"]
        self.top = kwargs["top"]
        self.l = kwargs["l"]

        self.NMI = readNMI(kwargs["NMI"], kwargs["n"], kwargs["l"])
        self.dis = readDis(kwargs["dis"], kwargs["n"])


    def sum_of_top(self, new, l):
        values = []
        size=len(new)
        for i in range(size):
            values.append(self.NMI[i][l])
        values.sort(reverse=True)
        value = 0
        for i in range(min(self.top, size)):
            value += values[i]
        return value

    def evaluateObjective(self, temp, new_item):
        g_old=temp[1]
        g_new=0
        new=temp[0]+[new_item]
        for l in range(self.l):
            g_new += self.sum_of_top(new, l)

        div = 0
        for i in range(len(temp[0])):
             div += self.dis[temp[0][i]][new_item]
        res = 0.5 * (g_new-g_old) + self.mylambda * div
        return res,g_new

    def select_best_item(self,res, items):
        evaluations=[]
        gs=[]
        for s in items:
            print("len:"+str(len(res[0]))+" new item:"+str(s))
            res1,g=self.evaluateObjective(res, s)
            evaluations+=[res1]
            gs+=[g]
        index = np.argmax(evaluations)
        return (res[0] + [items.pop(index)],gs[index]),items

    def greedy(self):
        items=[]
        res=([],0)
        for i in range(self.n):
            items.append(i)
        while len(res[0]) < self.k:
            print("constraint="+str(self.k)+" len(res)="+str(len(res[0])))
            new_res,left_items = self.select_best_item(res, items)
            items=left_items
            res=new_res
        return res

    def Calculate_true_value(self, res):
        size = len(res[0])
        g = 0
        new = res[0]
        for l in range(self.l):
            g += self.sum_of_top(new, l)

        div = 0
        for i in range(size):
            for j in range(i + 1, size):
                div += self.dis[res[0][i]][res[0][j]]
        res = g + self.mylambda * div

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

    def LocalSearch(self, init_res):
        items = []
        res = init_res
        for i in range(self.n):
            items.append(i)
        for i in init_res[0]:
            items.pop(items.index(i))
        # start with greedy solution
        while True:
            flag, value, res, exchang_item = self.exchange(res, items)
            if flag == False:
                break
            else:
                items.pop(items.index(exchang_item[0]))  # delete exchange_item[0] from V\X
                items.append(exchang_item[1])  # add exchang_item[1] in V\X
        return res, value

def OutputResult(res,value,log):
    log.write(str(value))
    log.write("\n")
    for item in res:
        log.write(str(item+1))
        log.write(' ')
    log.write("\n")
    log.close()


def run_experiments(args):
    problem = Problem(NMI=args.NMI, dis=args.dis, n=args.n_items, k=args.constraint, mylambda=args.mylambda,
                        top=args.top, l=args.l)
    print("doing greedy...")
    res_greedy = problem.greedy()
    value= problem.Calculate_true_value(res_greedy)

    print("saving greedy...")
    save_address = args.save_file + "/" + str(args.constraint)
    if not os.path.exists(save_address):
        os.makedirs(save_address)

    log = open(save_address + "/Result_greedy_" + str(args.constraint)+".txt", "w")
    OutputResult(res_greedy[0], value, log)

    print("doing LS...")
    res_ls, value = problem.LocalSearch(res_greedy)
    print("saving LS...")
    log = open(save_address + "/Result_LS_" + str(args.constraint)+".txt", "w")
    OutputResult(res_ls[0], value, log)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-arff', default='enron/enron.arff', type=str)
    argparser.add_argument('-xml', default='', type=str)
    argparser.add_argument('-p', '--save_file', help="save_result", default='enron_Result_Greedy_LS', type=str)
    argparser.add_argument('-n_items', help="number of features", default=1001, type=int)
    argparser.add_argument('-NMI', default='enron/NMI.txt', type=str)
    argparser.add_argument('-dis', default='enron/dis.txt', type=str)
    argparser.add_argument('-constraint', help="max size of subset", default=15, type=int)
    argparser.add_argument('-mylambda', help="trade_off", type=float, default=0.5)
    argparser.add_argument('-top', type=int, default=10)
    argparser.add_argument('-l', help="number of lables", default=53, type=int)
    argparser.add_argument('-dim', default=1702, type=int)
    args = argparser.parse_args()
    args.save_file = f'enron_Result_Greedy_LS/{args.mylambda}'
    run_experiments(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
