# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:09:22 2020

@author: Christal Chu
"""

import numpy as np
import nltk
# If you already have this package, hashtag nltk.download("aver..")
#nltk.download("averaged_perceptron_tagger")

alpha = 0.25
iteration = 10000

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
    LCS_matrix = [[0 for k in range(n+1)] for l in range(m+1)] 
    for i in range(m+1): 
        for j in range(n+1): 
            if(i==0 or j==0): 
                LCS_matrix[i][j] = 0
            elif(A[i-1] == B[j-1]): 
                LCS_matrix[i][j] = LCS_matrix[i-1][j-1] + 1
                max_num = max(max_num, LCS_matrix[i][j]) 
            else: 
                LCS_matrix[i][j] = 0      
    return max_num

def MCS_p(A, B, m, n):
    MCS_matrix = [[None]*(n+1) for i in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                MCS_matrix[i][j] = 0
            elif A[i-1] == B[j-1]:
                MCS_matrix[i][j] = MCS_matrix[i-1][j-1] + 1
            else:
                MCS_matrix[i][j] = max(MCS_matrix[i-1][j], MCS_matrix[i][j-1])
    return MCS_matrix[m][n]

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

def SquareError(X, cost):
    m = np.shape(X)[0]
    sumCost = np.sum(cost)
    J = sumCost/m 
    return J

def Cost(Hw, y):
    cost = -(np.multiply(y, np.log(Hw + 1e-6))) - np.multiply((1-y), np.log(1-Hw + 1e-6))
    return cost

def Hypo(X, w):
    X_w = np.dot(X, w)
    hypo = 1/(1+np.exp(-X_w))
    return hypo

def new_weights(w, Hw, X, y):
    m = np.shape(X)[0]
    new_w = w - (np.multiply(alpha/m, np.dot((Hw - y).T, X))).T
    return new_w

def line_prob(X, final_w):
    Hw = Hypo(X, final_w)
    return Hw


# train dataset
num_lines = sum(1 for line in open("quora_train.txt", "r", encoding = "unicode-escape"))-1
print("training dataset: ",num_lines) #7996 rows
filename = open("quora_train.txt", "r", encoding = "unicode-escape")
fline = filename.readline()
trainX = np.ones((num_lines, 6)) # (m, n) m: num of lines, n: 5 features+1
trainy = np.zeros((num_lines, 1)) # assign is_match to trainy
weights = np.full((6, 1), 0.1667) # n = 5 features+1, weights = 0.167
print("Initial weights: \n", weights)
match, unmatch = 0, 0
for i in range(0, num_lines):
    fline = filename.readline()
    is_match = int(fline[:1])
    #print(is_match)
    if is_match == 1:
        match += 1
    else:
        unmatch += 1
    fline = cleantext(fline[1:]).strip("\n").split("\t")
    #print(i, fline)
    A = fline[0]
    B = fline[1]
    A = A.split()
    B = B.split()
    m = len(A)
    n = len(B)
    JC = JC_p(A, B)
    CNS = CNS_p(A, B)
    MCS = MCS_p(A, B, m, n)/max(m, n)
    LCS = LCS_p(A, B, m, n)/max(m, n)
    UniG = len(NGram(A, B, 1))/max(m, n)
    BiG = len(NGram(A, B, 2))/max(m, n)
    TriG = len(NGram(A, B, 3))/max(m, n)
    WNGO = (UniG + BiG + TriG)/3
    trainX[i][1:] = (JC, CNS, LCS, MCS, WNGO)
    trainy[i] = is_match
filename.close()

Hw = Hypo(trainX, weights)
new_w = new_weights(weights, Hw, trainX, trainy) # new weights
cost = Cost(Hw, trainy)
J = SquareError(trainX, cost)
final_w = np.empty(np.shape(new_w))
for i in range(1, iteration):
    new_Hw = Hypo(trainX, new_w)
    new_cost = Cost(new_Hw, trainy)
    new_w = new_weights(new_w, new_Hw, trainX, trainy)
    if(i == iteration-1):
        final_w = new_w
print("Final weights: \n", final_w)
# --------------------------------------------
# test dataset
num_lines2 = sum(1 for line in open("quora_test.txt", "r", encoding = "unicode-escape"))-1
print("test dataset: ", num_lines2) # 1999 rows
testname = open("quora_test.txt", "r", encoding = "unicode-escape")
tline = testname.readline()
testX = np.ones((num_lines2, 6)) # (m, n) m: num of lines, n: features
#testy = np.zeros((num_lines2-1, 1)) 
match2, unmatch2 = 0, 0
tp, tn, fp, fn = 0, 0, 0, 0
for i in range(0, num_lines2):
    tline = testname.readline()
    is_match2 = int(tline[:1])
    if is_match2 == 1:
        match2 += 1
    else:
        unmatch2 += 1
    tline = cleantext(tline[1:]).strip("\n").split("\t")
    A2 = tline[0]
    B2 = tline[1]
    A2 = A2.split()
    B2 = B2.split()
    m2 = len(A2)
    n2 = len(B2)
    JC2 = JC_p(A2, B2)
    CNS2 = CNS_p(A2, B2)
    LCS2 = LCS_p(A2, B2, m2, n2)/max(m2, n2)
    MCS2 = MCS_p(A2, B2, m2, n2)/max(m2, n2)
    UniG2 = len(NGram(A2, B2, 1))/max(m2, n2)
    BiG2 = len(NGram(A2, B2, 2))/max(m2, n2)
    TriG2 = len(NGram(A2, B2, 3))/max(m2, n2)
    WNGO2 = (UniG2 + BiG2 + TriG2)/3
    testX[i][1:] = (JC2, CNS2, LCS2, MCS2, WNGO2)
    prob = line_prob(testX[i], final_w)
    if(prob < 0.5):
        prob = 0
    else:
        prob = 1
    if(prob == 1 and is_match2 == 1):   # predict paired is also a paired
        tp += 1
    elif(prob == 0 and is_match2 == 0): # predict not paired is not a paired
        tn += 1
    elif(prob == 1 and is_match2 == 0): # predict paired is not paired
        fp += 1
    else:                               # predict not paired is a paired
        fn += 1
        
testname.close()


print("Total number of paraphrases in test dataset: ", match2)
print("Total number of non paraphrases in test dataset: ", unmatch2)
print("FP: ", fp, ", TP: ", tp, "\nFN: ", fn, ", TN: ", tn)

Accuracy = (tp+tn)/(tp+tn+fp+fn)
Precision = (tp)/(tp+fp)
Recall = (tp)/(tp+fn)
F1 = 2/((1/Precision)+(1/Recall)) 

print("Accuracy: ", Accuracy, "\nPrecision: ", Precision, "\nRecall: ", Recall, "\nF1 score: ", F1)









