'''
you should fill the func given here
all the other imports/constants/classes/func should be stored here and only here (not in other files)
'''

############################
# Insert your imports here
############################
from collections import defaultdict
import numpy
import json
import pandas as pd
import copy
import os

def load_mempool_data(mempool_data_full_path, current_time=1510264253.0):
    with open(mempool_data_full_path) as f:
        mem_loaded= json.load(f)
        mem_data={}
        for transId in mem_loaded:
            if mem_loaded[transId]['time']<current_time and current_time< mem_loaded[transId]['removed']:
                mem_data[transId]=mem_loaded[transId]
    return mem_data



############################
# Part 1
############################

def greedy_knapsack(block_size, mempool_data):
    mem_list=[]
    for key in mempool_data:
        mempool_data[key]['transId']=key
        mempool_data[key]['ratio']= mempool_data[key]['fee']/mempool_data[key]['size']
    for key, value in mempool_data.items():
        mem_list.append(value)

    temp_size=0
    knapsack=[]
    greedy= pd.DataFrame(mem_list)
    sorted_greedy = greedy.sort_values('ratio',ascending= False, kind='mergesort')

    for row in sorted_greedy.itertuples():
        r_size=row[5]
        if r_size<=(block_size-temp_size):
            knapsack.append(row[7])
            temp_size+=r_size
    return knapsack



def evaluate_block(tx_list, mempool_data):
    revenue = 0.0
    for tx in tx_list:
        revenue= revenue + mempool_data[tx]['fee']
    return revenue

def VCG(block_size, tx_list, mempool_data):
    Vs=0.0
    Vs_j=0.0
    vcg={}
    basic_revenue= evaluate_block(tx_list,mempool_data)
    for tx in tx_list:
        Vs_j=(basic_revenue-mempool_data[tx]['fee'])
        mempool_data_no_tx= copy.deepcopy(mempool_data)
        del mempool_data_no_tx[tx]
        no_tx_list= greedy_knapsack(block_size, mempool_data_no_tx)
        Vs= evaluate_block(no_tx_list,mempool_data)
        vcg[tx]=Vs-Vs_j
        if vcg[tx]<0:
            vcg[tx]=0
    return vcg

	
############################
# Part 2
############################
	
def forward_bidding_agent(tx_size, value, urgency, mempool_data, block_size):
    t=600
    in_block=[]
    use_mempool = copy.deepcopy(mempool_data)
    utility= {}
    z=0
    z_last=5000
    GT={}
    for i in range (0,5010,10):
        GT[i]='null'
    while(len(use_mempool)>= 0):
        in_block=greedy_knapsack(block_size,use_mempool)
        if len(in_block)==0:
            break
        min_tx=in_block[-1]
        min_ratio=mempool_data[min_tx]['fee']/mempool_data[min_tx]['size']
        if min_ratio>5000:
            break
        while z <= z_last:
            if z >= min_ratio:
                GT[z] = t
            z=z+10
        z_last=min_ratio
        for id in in_block:
            del use_mempool[id]
        t += 600
    Gu={}
    z=0

    while (z<=5000):
        if GT[z]=='null':
            Gu[z]=0
        else:
            Gu[z] = ((value * 2 ** ((-1)*GT[z]) * urgency / 1000) - (z * tx_size))
        z+=10

    return max(Gu.keys(),key=lambda k: Gu[k]) #finds argmax



	

def truthful_bidding_agent(tx_size, value, urgency, mempool_data, block_size):

	z = value*(2**(-3.6*urgency))
	return z

