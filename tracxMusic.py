# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:48:08 2016

@author: pss02ca
"""

import tracx
import pandas as pd

tracx1 = tracx.Tracx()


def get_all_music_encodings():
    pitch_change = ["d","s","u"] #down,same,up
    loudness_change = ["e","l","q"] #equal,louder,quieter
    
    all_combinations = {}
    blank_encoding = [0] * 14
    all_combinations[" "]= list(blank_encoding)
    for i, pitch in enumerate(pitch_change):
        for duration in range(8):         
            for k, loud in enumerate(loudness_change):
                    token = pitch + str(duration+1) + loud
                    encoding = list(blank_encoding)
                    encoding[i]=1
                    encoding[3+duration] = 1
                    encoding[11+k]= 1                    
                    all_combinations[token] = list(encoding)
    return all_combinations
                    

def get_sequence_frequencies(music, length = 1):
    '''
    takes some music as a single list and sequence length and returns a 
    dictionary of all the sequences of given length and their frequency of
    occurance.
    '''
    all_seqs  = set()
    len_music = len(music)
    for i in range(0,len_music - length):
        # NB sets don't work on lists so convert to tuple
        all_seqs.add(tuple(music[i:i+length]))
    all_freqs = {}
    for seq in all_seqs:
        all_freqs[seq] = 0
        for i in range(0,len_music - length):
            if seq == tuple(music[i:i+length]):
                all_freqs[seq] += 1
    return all_freqs


#data = open('input.txt', 'r').read() # should be simple plain text file
#Yellow submarine Lennon & McCartney 1966 
ys = "u2q_u2l_d3q_d2l_u2q_d3l_s1q_d2l_d2q_d3l_s1q_u5l_s1q_d2l_u2l_u2l_d3q_d2e_u2l_d3l_s1q_d2l_d2q_d3l_s1q_u5l_s1q_d2l_u2q_u2l_d3q_d2l_u2q_d3l_s1q_d2l_d2q_d3l_s1q_u5l_s1q_d2l_u2q_u2l_d3q_d2l_u2q_d3l_s1q_d2l_d2q_d3l_u5q_s1q_d2l_s1e_s1e_s1q_u2q_d5l_s1q_s1l_s1q_s1l_s1q_s1l_s1q_s1l_d2q_s1q_s1l_s1q_s1l_s1e_s1e_s1q_u2q_d5l_s1q_s1l_s1q_s1l_s1q_s1l_s1q_s1l_d2q_s1q_s1l_s1q_s1l_s1e_s1e_s1q_u2q_d5l_s1q_s1l_s1q_s1l_s1q_s1l_s1q_s1l_s1q_s1l_s1q_s1l"
ys = ys.split("_")

#freq_item = get_sequence_frequencies(ys,1)
freq= {}
freq.update(get_sequence_frequencies(ys,2))
freq.update(get_sequence_frequencies(ys,3))
freq.update(get_sequence_frequencies(ys,4))
freq.update(get_sequence_frequencies(ys,5))


tracx1.set_training_data(ys)
tracx1.set_input_encodings(get_all_music_encodings())

print(tracx1.get_input_encoding("s1q"))
print(tracx1.get_input_encoding("u2l"))
print(tracx1.get_unique_items())  

print(tracx1.inputWidth)

#print(tracx1.decimal_to_binary(101))
#
tracx1.initialize_weights()
#
##print(tracx1.layer[0])
#
ret = tracx1.run_full_simulation()
#
#print(ret) 
#
#ret = tracx1.test_strings(["abc","def,ghi"] )
#print(ret)

all_results = pd.DataFrame(columns = ["seq","len","freq","totalDelta","meanDelta"])

for item in freq:
    ret = tracx1.test_string(list(item))
    all_results.loc[len(all_results)] = ["_".join(item),len(item),freq[item], ret["totalDelta"], ret["meanDelta"]]

print(all_results)   

all_results.to_csv("yellow_submarine_out.csv") 