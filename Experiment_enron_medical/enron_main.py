import os
import random
import argparse
import sys
import enron_GSEMO as prd
import numpy as np

import time, datetime
def OutputResult(p,res,log):
    final_value=0
    for info in res:
        final_value=p.Calucalate_true_value(info)
        log.write(str(final_value))
        log.write("\n")

        index = np.nonzero(info)

        linklist = []
        for i, j in zip(index[0], index[1]):
            linklist.append([i, j])

        for item in linklist:
            log.write(str(item[1]+1))
            log.write(' ')
        log.write("\n")
    log.close()
    return final_value

def OutputAvg(all_value,log):
    log.write(str(np.mean(all_value)))
    log.close()


def run_experiments(args):
    save_address = args.save_file + "/" + str(args.constraint)
    if not os.path.exists(save_address):
        os.makedirs(save_address)
    problem = prd.GSEMO(NMI=args.NMI, dis=args.dis, n=args.n_items, k=args.constraint, mylambda=args.mylambda,
                        top=args.top, l=args.l)
    for i in range(args.times):
        path = save_address + "/Result_" + str(time.time()) + ".txt"
        result = problem.doPORD(path)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-arff', default='enron/enron.arff', type=str)
    argparser.add_argument('-xml', default='', type=str)
    argparser.add_argument('-NMI', default='enron/NMI.txt', type=str)
    argparser.add_argument('-dis', default='enron/dis.txt', type=str)
    argparser.add_argument('-times', default=10, type=int)
    argparser.add_argument('-p', '--save_file', help="save_result", default='enron_Result_PORD', type=str)
    argparser.add_argument('-n_items', help="number of features", default=1001, type=int)
    argparser.add_argument('-constraint', help="max size of subset", default=50, type=int)
    argparser.add_argument('-mylambda', help="trade_off", type=float, default=0.5)
    argparser.add_argument('--top', type=int, default=10)
    argparser.add_argument('-l', help="number of lables", default=53, type=int)
    argparser.add_argument('-dim', default=1702, type=int)
    args = argparser.parse_args()
    args.save_file = f'enron_Result_GSEMO/{args.mylambda}'
    run_experiments(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
