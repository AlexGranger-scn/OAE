import torch
import os
import argparse
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance
import pdb
import torch.backends.cudnn as cudnn
import sys
from model import OrderedAutoEncoder
import scipy
from sklearn.utils.extmath import svd_flip
import time 
from functools import wraps
import re

def timing(f):
    """print time used for function f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        t = time.time() - time_start
        print(f'total time = {t:.4f}')
        return ret

    return wrapper

def transfer_format(input_sr_path, output_sr_path, dataset):
    # pdb.set_trace()
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
        elif re.match('cnc', dataset):
            enroll_label = lst[0].split('-')[0]
            f.write("{}".format(enroll_label))
            for i in range(1, len(lst)):
                test_label = lst[i].split('-')[0]
                f.write(" {}".format(test_label))
            f.write('\n')
    f.close()

def calAllScore(path_csv,enroll_emb,val_emb):
    print("----------calculating score------------")

    with open(path_csv, 'w', encoding='UTF8', newline='') as f:
        header = ['spk-id', 'utt-id', 'scores']
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        pbar = tqdm(total=len(enroll_emb))
        for e in enroll_emb:
            writer = csv.writer(f)
            e_emb = enroll_emb[e]
            
            for v in val_emb:
                data = []
                data.append(e)
                v_emb = val_emb[v]
                data.append(v)
                e_emb = torch.sign(e_emb.reshape(-1)).detach().type(torch.int)
                v_emb = torch.sign(v_emb.reshape(-1)).detach().type(torch.int)
                score = 1-distance.hamming(e_emb,v_emb)
                data.append(score)
                writer.writerow(data)               
            pbar.update(1)
        pbar.close()

def selectTop(path_csv,path_top):

    df = pd.read_csv(path_csv)
    df = df.groupby('spk-id').apply(lambda x:x.nlargest(int(10),'scores'))
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

def cal_topk(input_sr_path):
    f = open(input_sr_path, 'r')

    lines = f.readlines()
    f.close()

    num_spk = 0
    top_1=0
    top_3=0
    top_5=0
    top_10=0
    
    for line in lines:
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
    

    print("top1_accuracy = %.3f"%top_1) 
    print("top3_accuracy = %.3f"%top_3) 
    print("top5_accuracy = %.3f"%top_5) 
    print("top10_accuracy = %.3f"%top_10) 

        
    return top_1,top_3,top_5,top_10

@timing
def topk(score_csv,test,enroll,path_top,output_sr_path,data_set):
    calAllScore(score_csv,test,enroll)
    selectTop(score_csv,path_top)
    transfer_format(path_top,output_sr_path, data_set)
    top_1,top_3,top_5,top_10 = cal_topk(output_sr_path)

    return top_1,top_3,top_5,top_10 

def divide_emb(emb,result):
    i=0
    for en in emb:
        emb[en]=result[i]
        i+=1
    return emb

def choose_gpu(i_gpu):
    """choose current CUDA device"""
    torch.cuda.device(i_gpu).__enter__()
    cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--enroll_path',type=str,default=None, help='path of enroll x-vector')
    parser.add_argument('--test_path',type=str,default=None, help='path of test x-vector')
    parser.add_argument('--enroll_matrix',type=str,default=None, help='matrix of enroll vector')
    parser.add_argument('--test_matrix',type=str,default=None, help='matrix of test vector')
    parser.add_argument('--cut_dimension',type=int,default=None, help='the test fragment')

    parser.add_argument('--score_csv', type=str, default=None, help='path of raw sid result')
    parser.add_argument('--path_top', type=str, default=None, help='top10 metadata dir')
    parser.add_argument('--output_sr_path', type=str, default=None, help='path of meta result')

    parser.add_argument('--binary',type=int,default=None, help='length of hidden layer')
    parser.add_argument('--ngpu',type=int,default=2, help='choose a gpu')

    parser.add_argument('--pth_path',type=str,default=None, help='path of the checkpoint')
    parser.add_argument('--data_set',type=str,default=None, help='the test dataset')
    parser.add_argument('--log_path',type=str,default=None, help='')

    args = parser.parse_args()
    
    #cuda
    choose_gpu(args.ngpu)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('Training on device{}'.format(device))

    state_dict = torch.load(args.pth_path, map_location="cpu")

    data_enroll_dict = np.load(args.enroll_path,allow_pickle=True).item()
    data_test_dict = np.load(args.test_path,allow_pickle=True).item()

    model = OrderedAutoEncoder(args.binary)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(model)

    with torch.no_grad():

        print("----------matrix emb-------------")
        enroll_emb = np.load(args.enroll_matrix,allow_pickle=True)
        test_emb = np.load(args.test_matrix,allow_pickle=True)
        print("----------done-------------")


        print("----------model-------------")
        enroll_emb,_ = model(torch.tensor(enroll_emb))
        test_emb,_ = model(torch.tensor(test_emb))
        print("----------done-------------")

        enroll_emb = enroll_emb[:,0:args.cut_dimension]
        test_emb = test_emb[:,0:args.cut_dimension]
        print(enroll_emb.shape)

        print("----------divide emb-------------")
        enroll = divide_emb(data_enroll_dict,enroll_emb )
        test = divide_emb(data_test_dict,test_emb )
        print("----------done-------------")
        
        top_1,top_3,top_5,top_10 =topk(args.score_csv,test,enroll,args.path_top, args.output_sr_path, args.data_set)

        f = open(os.path.join(args.log_path,"topn.log"), 'a')
        f.write("cut-dimension: "+ str(args.cut_dimension)+'\n')
        f.write("top1: "+'{:.3f}'.format(top_1) +'\n')
        f.write("top3: "+'{:.3f}'.format(top_3) +'\n')
        f.write("top5: "+'{:.3f}'.format(top_5) +'\n')
        f.write("top10: "+'{:.3f}'.format(top_10) +'\n')

        f.close()
