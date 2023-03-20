import matplotlib.pyplot as plt
#import networkx as nx
import torch
import argparse
import numpy as np
import pdb
from model import *
import os
import csv
import time 
from functools import wraps
from scipy.spatial import distance
import sys
import pandas as pd

class Node():
    def __init__(self, item, count):
        self.item = item
        self.lchild = None
        self.rchild = None
        self.id = []
        self.count = count
        self.layer = (count-1) % args.cut_dimension
    def add_id (self, label):
        self.id.append(label)

class Tree():
    def __init__(self):
        self.root = Node(0,0)

    def add(self, emb, label,count):
        cur = self.root

        for item in emb:
            node = Node(item,count)
            count += 1
            if item == -1:
                if cur.lchild == None:
                    cur.lchild = node
                cur = cur.lchild
            elif item == 1:
                if cur.rchild == None:
                    cur.rchild = node
                cur = cur.rchild
            cur.add_id(label)


    def leaf(self,root):
        if root==None:
            return 0 
        elif root.lchild==None and root.rchild==None:
            return 1 
        else:
            return (self.leaf(root.lchild)+self.leaf(root.rchild))  
    
    def match_all_length(self, root, test_emb):
        cur = root
        for i in test_emb:
            if i == -1 :
                cur = cur.lchild
            else :
                cur = cur.rchild
        return cur.id[0]
    
    def match_all_length_record(self, root, test_emb):
        cur = root
        for i in test_emb:
            if i == -1 :
                if cur.lchild is not None:
                    cur = cur.lchild
                else:
                    return "pass"
            else :
                if cur.rchild is not None:
                    cur = cur.rchild
                else:
                    return "pass"
        return cur.id[0]
    

def timing(f):
    """print time used for function f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        t = time.time() - time_start
        print(f'total time = {t:.4f}'+'\n')
        return ret

    return wrapper

def divide_emb(emb,result):
    i=0
    new_emb = {}
    
    for en in emb:
        spk_id = en
        new_emb[spk_id]=result[i]
        i+=1
    return new_emb

def match_all_length_record(tree, test):
    acc = 0
    num_spk = 0
    num_spk_total = 0
    test_id_list_hamming=[]
    test_id_list_hamming_key=[]
    for t in test:
        id_lst= tree.match_all_length_record(tree.root,test[t].type(torch.int).numpy())
        num_spk_total += 1 
        if args.dataset == 'vox':
            test_id = t.split('/')[7]
        elif args.dataset == 'cnc':
            test_id = t. split('-')[0]
        #print(d)
        if id_lst!='pass':
            num_spk += 1
            if test_id in id_lst:
                acc += 1
            test_id_list_hamming.append(test[t])
            test_id_list_hamming_key.append(t)
    acc_m = acc / num_spk
    acc_total = acc / num_spk_total

    print("match_all_length_num = %.3f"%num_spk) 
    print("match_all_length_accuracy = %.3f"%acc_m) 
    print("match_all_length_accuracy_total = %.3f"%acc_total) 
    return num_spk, acc, test_id_list_hamming, test_id_list_hamming_key

@timing
def match_all_length(tree, test):
    tree.match_all_length(tree.root,test.type(torch.int).numpy())


def calAllScore(path_csv,enroll_emb,test_emb_list, test_emb):
    print("----------calculating score------------")

    with open(path_csv, 'w', encoding='UTF8', newline='') as f:
        header = ['spk-id', 'utt-id', 'scores']
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        for t in test_emb_list:
            writer = csv.writer(f)
            t_emb = test_emb[t]
            
            for e in enroll_emb:
                data = []
                data.append(t)
                e_emb = enroll_emb[e]
                data.append(e)
                score = 1-distance.cosine(t_emb,e_emb)
                #print(score)
                data.append(score)
                writer.writerow(data)  
                
def calAllScore_hamming(path_csv,enroll_emb,test_emb_list, test_emb):
    print("----------calculating score------------")

    with open(path_csv, 'w', encoding='UTF8', newline='') as f:
        header = ['spk-id', 'utt-id', 'scores']
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        for t in test_emb_list:
            writer = csv.writer(f)
            t_emb = test_emb[t]
            
            for e in enroll_emb:
                data = []
                data.append(t)
                e_emb = enroll_emb[e]
                data.append(e)
                score = 1-distance.cosine(t_emb,e_emb)
                #print(score)
                data.append(score)
                writer.writerow(data)   

def selectTop(path_csv,path_top):
    #header = ['spk-id', 'utt-id', 'scores']
    df = pd.read_csv(path_csv)
    df = df.groupby('spk-id').apply(lambda x:x.nlargest(int(10),'scores'))
    #data = 
    with open(path_top, 'w') as f:
        old_spk = ' '
        for index, row in df.iterrows():
            spk = row['spk-id']
            if old_spk == ' ':
                old_spk = spk
                f.write(spk)
            elif old_spk != spk:
                old_spk = spk
                f.write('\n' + spk)
            f.write(' ' + row['utt-id'])

def transfer_format(input_sr_path, output_sr_path, dataset):
    f = open(input_sr_path, 'r')
    input_lines = f.readlines()
    f.close()

    f = open(output_sr_path, 'w')
    for line in input_lines:
        lst = line.strip().split(' ')
        if dataset=="vox":
            enroll_path = lst[0]
            enroll_label = enroll_path.split('/')[7]
            f.write("{}".format(enroll_label))
            for i in range(1, len(lst)):
                test_label = lst[i].split('/')[7]
                f.write(" {}".format(test_label))
            f.write('\n')
        elif dataset=="cnc":
            enroll_label = lst[0].split('-')[0]
            f.write("{}".format(enroll_label))
            for i in range(1, len(lst)):
                test_label = lst[i].split('-')[0]
                f.write(" {}".format(test_label))
            f.write('\n')
    f.close()

def cal_topk(input_sr_path):
    f = open(input_sr_path, 'r')
    #pdb.set_trace()
    lines = f.readlines()
    f.close()

    num_spk = 0
    top_1=0
    top_3=0
    top_5=0
    top_10=0
    for line in lines:
        #print(line)
        num_spk += 1
        lst = line.strip().split(' ')
        spk_id = lst[0].split('-')[0]
        for i in range(1, len(lst)):
            utt_id = lst[i]
            if utt_id == spk_id:
                if i==1:
                    top_1 += 1
                if i<=3:
                    top_3 += 1
                if i<=5:
                    top_5 += 1
                if i<=10:
                    top_10+=1

    top_1 /= num_spk
    top_3 /= num_spk
    top_5 /= num_spk
    top_10 /= num_spk
    
    print("mae-desen-num-spk = %.3f"%num_spk)
    print("top1_accuracy = %.3f"%top_1) 
    print("top3_accuracy = %.3f"%top_3) 
    print("top5_accuracy = %.3f"%top_5) 
    print("top10_accuracy = %.3f"%top_10) 

        
    return top_1,top_3,top_5,top_10



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enroll_path',type=str,default=None)
    parser.add_argument('--test_path',type=str,default=None)
    parser.add_argument('--enroll_matrix',type=str,default=None)
    parser.add_argument('--test_matrix',type=str,default=None)
    parser.add_argument('--cut_dimension',type=int,default=None)
    parser.add_argument('--dataset',type=str,default=None)
    
    parser.add_argument('--binary',type=int,default=None)
    parser.add_argument('--ngpu',type=int,default=2)

    parser.add_argument('--pth_path_binary',type=str,default=None)
    parser.add_argument('--pth_path_dense',type=str,default=None)
    parser.add_argument('--fig_path',type=str,default=None)
    parser.add_argument('--mean_depth',type=int,default=None)
    
    parser.add_argument('--csv_path',type=str,default=None)
    parser.add_argument('--path_top',type=str,default=None)
    parser.add_argument('--output_sr_path',type=str,default=None)
    parser.add_argument('--stage',type=int,default=None)

    args = parser.parse_args()
    count =1

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('Training on device{}'.format(device))

    state_dict = torch.load(args.pth_path_binary, map_location="cpu")
    state_dict_dense = torch.load(args.pth_path_dense, map_location="cpu")

    data_enroll_dict = np.load(args.enroll_path,allow_pickle=True).item()
    data_test_dict = np.load(args.test_path,allow_pickle=True).item()

    model = OrderedAutoEncoder(args.binary)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    model_dense = OrderedAutoEncoder(args.binary)
    model_dense.load_state_dict(state_dict_dense, strict=False)
    print(model_dense)
    
    with torch.no_grad():

        print("----------matrix emb-------------")
        enroll_emb = np.load(args.enroll_matrix,allow_pickle=True)
        test_emb = np.load(args.test_matrix,allow_pickle=True)
        print("----------done-------------")

        print("----------model_binary-------------")
        enroll_emb_binary,_ = model(torch.tensor(enroll_emb))
        test_emb_binary,_ = model(torch.tensor(test_emb))
        print("----------done-------------")

        print("----------model_dense-------------")
        enroll_emb_dense,_ = model_dense(torch.tensor(enroll_emb))
        test_emb_dense,_ = model_dense(torch.tensor(test_emb))
        print("----------done-------------")

        enroll_emb_binary = enroll_emb_binary[:,0:args.cut_dimension]
        test_emb_binary = test_emb_binary[:,0:args.cut_dimension]

        enroll_emb_dense = enroll_emb_dense[:,0:args.cut_dimension]
        test_emb_dense = test_emb_dense[:,0:args.cut_dimension]

        enroll_emb_binary = torch.sign(enroll_emb_binary).detach().type(torch.int)
        test_emb_binary = torch.sign(test_emb_binary).detach().type(torch.int)

        print("----------divide emb-------------")
        enroll_binary = divide_emb(data_enroll_dict,enroll_emb_binary)
        test_binary = divide_emb(data_test_dict,test_emb_binary)

        enroll_dense = divide_emb(data_enroll_dict,enroll_emb_dense)
        test_dense = divide_emb(data_test_dict,test_emb_dense)
        print("----------done-------------")

        # store the enroll data in a binary tree
        tree = Tree()
        for en in enroll_binary:
            if args.dataset == 'vox':
                tree.add(enroll_binary[en].numpy(), en.split('/')[7],count)
                count += args.cut_dimension
            elif args.dataset == 'cnc':
                tree.add(enroll_binary[en].numpy(), en.split('-')[0],count)
                count += args.cut_dimension
        
        # number of leaf nodes
        num = tree.leaf(tree.root)
        print("number of leaf nodes: "+ str(num))


        # match and record mean depth
        num_spk, acc, test_list, test_list_dict = match_all_length_record(tree,test_binary)
        
        calAllScore(args.csv_path, enroll_dense,test_list_dict,test_dense)
        selectTop(args.csv_path,args.path_top)
        transfer_format(args.path_top, args.output_sr_path, args.dataset)
        top_1,top_3,top_5,top_10 = cal_topk(args.output_sr_path)


        calAllScore_hamming(args.csv_path, enroll_binary,test_list_dict,test_binary)
        selectTop(args.csv_path,args.path_top)
        transfer_format(args.path_top, args.output_sr_path, args.dataset)
        top_1h,top_3h,top_5h,top_10h = cal_topk(args.output_sr_path)

        if args.stage >= 0:
            path = "speed-test/test_result"+'/'+str(args.dataset)+'/'+'result_info.log'
            f = open(path, 'a')
            f.write("cut-dimension: "+ str(args.cut_dimension)+'\n')
            f.write('\t'+"Binary tree record: "+'\n')
            f.write('\t'+'\t'+"number of leaf node: "+'{:.5f}'.format(num) +'\n')
            f.write('\t'+'\t'+"number of speaker search all length: "+'{:.5f}'.format(num_spk) +'\n')
            f.write('\t'+'\t'+"accuracy of binary tree: "+'{:.5f}'.format(acc/num_spk) +'\n')
            f.write('\t'+"MAE dense result: "+'\n')
            f.write('\t'+'\t'+"dense MAE top1: "+'{:.5f}'.format(top_1) +'\n')
            f.write('\t'+'\t'+"dense MAE top3: "+'{:.5f}'.format(top_3) +'\n')
            f.write('\t'+'\t'+"dense MAE top5: "+'{:.5f}'.format(top_5) +'\n')
            f.write('\t'+'\t'+"dense MAE top10: "+'{:.5f}'.format(top_10) +'\n')
            f.write('\t'+"Hamming accuracy: "+'\n')
            f.write('\t'+'\t'+"hamming top1: "+'{:.5f}'.format(top_1h) +'\n')
            f.write('\t'+'\t'+"hamming top3: "+'{:.5f}'.format(top_3h) +'\n')
            f.write('\t'+'\t'+"hamming top5: "+'{:.5f}'.format(top_5h) +'\n')
            f.write('\t'+'\t'+"hamming top10: "+'{:.5f}'.format(top_10h) +'\n')

        if args.stage >= 1:
            path = "speed-test/test_result"+'/'+str(args.dataset)+'/'+'result_info.log'
            f = open(path, 'a')

            if args.stage >= 1:
                st_t = time.time()
                for j in range(10):
                    match_all_length(tree,test_list[0])
                tree_time = (time.time()-st_t) /(10)
                f.write("Time of tree: "+'{:.5f}'.format(tree_time) +'\n'+'\n') 

            if args.stage >= 2:
                st_c = time.time()
                for j in range (10):
                        for ed in enroll_dense:
                                distance.cosine(test_dense[test_list_dict[0]],enroll_dense[ed])
                cosine_time = (time.time()-st_c) /(10)
                f.write("Time of cosine: "+'{:.5f}'.format(cosine_time) +'\n')    

            if args.stage >= 3:  
                st_h = time.time()
                for i in range(10):
                    for eb in enroll_binary:
                        distance.hamming(test_list[0],enroll_binary[eb])
                hamming_time = (time.time()-st_h) /(10)
                f.write("Time of hamming: "+'{:.5f}'.format(hamming_time) +'\n') 
        
