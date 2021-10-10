import os
import random
import argparse
import sys
import numpy as np
import math

global H_feature,H_label,H_feature_lable
H_feature={}
H_label={}
H_feature_lable={}


global feature_item,label_num
feature_item=1001
label_num=53

def read_sparse_arff(f_path, xml_path):
    global feature_item
    f = open(f_path)
    f_data = f.read().split('@data')
    f.close()
    column = [i.split(' ')[1] for i in f_data[0].split('@attribute')[1:]]
    label_row, label_col, label_data, label_indptr = [], [], [], []
    feature_row, feature_col, feature_data, feature_indptr = [], [], [], []
    inil = 0
    inil_label = 0
    for l in enumerate(f_data[1].replace(' ', ':').split('\n')[1:-1]):
        l_v_dict = eval(l[1])
        col, data = [], []
        col.extend(l_v_dict.keys())
        data.extend(l_v_dict.values())
        index = 0
        feature_num = 0
        for item in col:
            if item < feature_item:
                index += 1
                feature_num += 1
            else:
                break
        inil_label += len(l_v_dict) - feature_num
        label_indptr.append(inil_label)
        label_col.extend(col[index:])
        label_data.extend(data[index:])
        label_row.extend([l[0] for i in range(len(l_v_dict) - feature_num)])

        inil += feature_num
        feature_indptr.append(inil)
        feature_col.extend(col[0:index])
        feature_data.extend(data[0:index])
        feature_row.extend([l[0] for i in range(feature_num)])

    sparse_feature = (feature_data, feature_row, feature_col, feature_indptr)
    sparse_label = (label_data, label_row, label_col, label_indptr)

    return sparse_feature, sparse_label

def JointEntropy(distributionXY):
    je = 0
    [lenY, lenX] = np.shape(distributionXY)
    for i in range(lenY):
        for j in range(lenX):
            if (distributionXY[i][j] != 0):
                je = je - distributionXY[i][j] * math.log2(distributionXY[i][j])
    return je

def OutputNMI(feature, label, log):
    global feature_item, label_num
    print("computing and saving NMI(feature,lable):")
    for i in range(feature_item):
        for l in range(label_num):
            print("i = "+str(i)+" l = "+str(l))
            nmi=NMI(feature, label, i, l, args.dim)
            nmi=float('%.5f' % nmi)
            print(nmi)
            log.write(str(nmi))
            log.write(' ')
        log.write("\n")
    log.close()

def DataEntropy(dataArrayLen, diffDataNum):
    diffDataArrayLen = len(diffDataNum)
    entropyVal = 0;
    p=[]
    for i in diffDataNum:
        proptyVal = diffDataNum[i] / dataArrayLen
        p.append(proptyVal)
        if (proptyVal != 0):
            entropyVal = entropyVal - proptyVal * math.log2(proptyVal)
    return entropyVal

def NMI(feature,label,i,j,true_len):
    global feature_item
    add = feature_item
    j = j + add
    global H_feature,H_label,H_feature_lable
    if (i,j) in H_feature_lable:
        je=H_feature_lable[(i,j)]
    else :
        # diffDataCount_X is the horizontal component is distribution table
        diffDataCount_X = true_len

        # diffDataCount_Y is the vertical component is distribution table
        diffDataCount_Y = true_len

        distributionXY = np.zeros((diffDataCount_Y, diffDataCount_X))

        before_f = 0
        before_l=0
        indptr_f = feature[3]
        indptr_l = label[3]
        diffDataX = []
        diffDataNumX={}
        diffDataNumY={}
        diffDataY = []
        for (index_f,index_l) in zip(indptr_f,indptr_l):

            data_f = feature[0][before_f:index_f]
            col_f = feature[2][before_f:index_f]
            before_f = index_f

            data_l = label[0][before_l:index_l]
            col_l = label[2][before_l:index_l]

            before_l = index_l
            i_exit = col_f.count(i)#feature i
            j_exit = col_l.count(j)#label j

            if (i_exit > 0):
                data_i = data_f[col_f.index(i)]
            else:
                data_i = 0

            if diffDataX.count(data_i) > 0:
                distribution_x = diffDataX.index(data_i)
                diffDataNumX[data_i]+=1
            else:
                diffDataX.append(data_i)
                distribution_x = len(diffDataX) - 1
                diffDataNumX[data_i] =1

            if (j_exit > 0):
                data_j = data_l[col_l.index(j)]
            else:
                data_j = 0

            if diffDataY.count(data_j) > 0:
                distribution_y = diffDataY.index(data_j)
                diffDataNumY[data_j] += 1
            else:
                diffDataY.append(data_j)
                distribution_y = len(diffDataY) - 1
                diffDataNumY[data_j] =1

            distributionXY[distribution_x][distribution_y] += 1

        diffDataCount_X = len(diffDataX)
        diffDataCount_Y = len(diffDataY)

        distributionXY = distributionXY[:diffDataCount_X, :diffDataCount_Y]
        distributionXY = distributionXY / true_len
        je = JointEntropy(distributionXY)
        H_feature_lable[(i, j)]=je
    if i in H_feature:
        HX=H_feature[i]
    else:
        HX = DataEntropy(true_len, diffDataNumX)
        H_feature[i]=HX
    if j in H_label:
        HY=H_label[j]
    else:
        HY = DataEntropy(true_len, diffDataNumY)
        H_label[j]=HY

    if (HX == 0.0 or HY == 0.0):
        return 0
    mi=HX+HY-je
    res = mi / math.sqrt(HX * HY)
    return res

def operate(args):
    feature,label=read_sparse_arff(args.arff, args.xml)
    save_address = args.s
    if not os.path.exists(save_address):
        os.makedirs(save_address)
    feature, lable = read_sparse_arff(args.arff, args.xml)
    save_address = args.s
    if not os.path.exists(save_address):
        os.makedirs(save_address)

    log = open(save_address + "/NMI.txt", "w")
    OutputNMI(feature, lable, log)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-arff', default='enron/enron.arff', type=str)
    argparser.add_argument('-xml', default='', type=str)
    argparser.add_argument('-s', help="save_file", default='enron', type=str)
    argparser.add_argument('-dim', default=1702, type=int)
    args = argparser.parse_args()
    operate(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
