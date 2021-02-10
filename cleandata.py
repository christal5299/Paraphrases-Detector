# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:09:22 2020

@author: Christal Chu
"""

import numpy as np
import nltk
# If you already have this package, hashtag nltk.download("aver..")
#nltk.download("averaged_perceptron_tagger")

alpha = 0.1
iteration = 1000

def cleantext(text):
    text = text.lower()
    text = text.strip()
    for i in text:
        if(i in """[]!.,"-!-@;':#$%^&*()+/?"""):
            text = text.replace(i, " ")
    return text

def JC_p(A, B):
    A = set(A)
    B = set(B)
    JC = len(A.intersection(B))/len(A.union(B))
    return JC

def find_nouns(S):
    tag = nltk.pos_tag(S)
    a = list()
    a = [tag[i][0] for i in range(len(tag)) if tag[i][1] == "NN" or tag[i][1] == "NNS"]
    return a
    
def CNS_p(A, B):
    ACN = set(find_nouns(A))
    BCN = set(find_nouns(B))
    if(max(len(ACN), len(BCN)) == 0):
        return 0
    CNS = len(ACN.intersection(BCN))/max(len(ACN), len(BCN))
    return CNS

def LCS_p(A, B, m, n):
    max_num = 0
    # Create a table to store lengths of 
    # longest common suffixes of substrings.  
    # Note that LCSuff[i][j] contains the  
    # length of longest common suffix of  
      
    # LCSuff is the table with zero  
    # value initially in each cell 
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)] 
  
    # Following steps to build 
    # LCSuff[m+1][n+1] in bottom up fashion 
    for i in range(m + 1): 
        for j in range(n + 1): 
            if (i == 0 or j == 0): 
                LCSuff[i][j] = 0
            elif (A[i-1] == B[j-1]): 
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                max_num = max(max_num, LCSuff[i][j]) 
            else: 
                LCSuff[i][j] = 0      
    return max_num

def MCS_p(A, B, m, n):
    L = [[None]*(n+1) for i in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif A[i-1] == B[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]

def intersection(list1, list2):
    return [item for item in list1 if item in list2]

def NGram(A, B, grams):
    a = []
    b = []
    cnt, cnt2 = 0, 0
    for token1 in A[:len(A)-grams+1]:
        a.append(A[cnt:cnt+grams])
        cnt += 1
    for token2 in B[:len(B)-grams+1]:
        b.append(B[cnt2:cnt2+grams])
        cnt2 += 1
    AB = intersection(a, b)
    return AB


# train dataset
num_lines = sum(1 for line in open("quora_train.txt", "r", encoding = "unicode-escape"))
# print(num_lines) 3997 rows
filename = open("quora_train.txt", "r", encoding = "unicode-escape")
fline = filename.readline()

trainX = np.ones((4, 5)) # (m, n) m: num of lines, n: features
trainy = np.full((4, 1), 0.5) # assume y = 0.5
weights = np.full((5, 1), 0.23) #n = features+1, weights = 0.5
print("Initial weights: \n", weights)
match, unmatch = 0, 0
for i in range(0, 4):
    fline = filename.readline()
    is_match = int(fline[:1])
    print(is_match)
    if is_match == 1:
        match += 1
    else:
        unmatch += 1
    fline = cleantext(fline[1:]).strip("\n").split("\t")
    print(fline)
    A = fline[0]
    B = fline[1]
    A = A.split()
    B = B.split()
    m = len(A)
    n = len(B)
    JC = JC_p(A, B)
    CNS = CNS_p(A, B)
    MCS = MCS_p(A, B, m, n)/max(m, n)
    LCS = LCS_p(A, B, m, n)#/max(m, n)
    print(LCS,"\n")
    #UniG = len(NGram(A, B, 1))/max(m, n)
    #BiG = len(NGram(A, B, 2))/max(m, n)
    #TriG = len(NGram(A, B, 3))/max(m, n)
    #WNGO = (UniG + BiG + TriG)/3
    #trainX[i][1:] = (JC, CNS, MCS, WNGO)
    
filename.close()









