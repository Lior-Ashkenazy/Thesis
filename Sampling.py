import random

import numpy as np
import matplotlib.pyplot as plt



# ----choose attention----
# find the function using binary search
def binarySearch(arr, x, low, high, z_arr):
    if low > high:
      return None
    else:
      mid = (low + high) // 2
      attentionSize_arr, average_attentionSize = get_average_attentionSize(arr[mid][0], arr[mid][1], z_arr)
      if x == average_attentionSize:
            return arr[mid] # the appropriate combunation
      elif x > average_attentionSize:
            return binarySearch(arr, x, mid + 1, high, z_arr)
      else:
            return binarySearch(arr, x, low, mid - 1, z_arr)


def combination_search(combinations, z_arr, average_attentionSize_fixed):
    low = 0
    high = len(combinations) - 1
    combination = binarySearch(combinations, average_attentionSize_fixed, low, high, z_arr)
    return combination

# Average attention size
def get_average_attentionSize(c,d,z_arr):
    attentionSize_arr = []
    point1, point2, point3 = [0,0], [c,d], [1,0]
    a1 = (point2[1]-point1[1])/(point2[0]-point1[0])
    b1 = point1[1]-(a1*point1[0])
    a2 = (point2[1]-point3[1])/(point2[0]-point3[0])
    b2 = point3[1]-(a2*point3[0])
    # print (a1, b1)
    # print(a2, b2)
    for zi in z_arr:
        if zi <= c: a, b = a1, b1
        else: a, b = a2, b2
        f_z = a*zi + b # f_z = attentionSize
        attentionSize_arr.append(f_z)
    attentionSize_arr = np.round(attentionSize_arr,3)
    average_attentionSize = sum(attentionSize_arr) / len(attentionSize_arr)
    average_attentionSize = round(average_attentionSize,1) # round 1 digits after the point
    return attentionSize_arr, average_attentionSize

def get_attention_labels(attention_size_arr ,votes_arr, voters_num):
    attention_label_arr_ALL = []
    for i in range(10):
        attention_label_arr = []
        for (attention_size, votes) in zip(attention_size_arr, votes_arr): # running on both arrays in parallel
            if attention_size>0:
                k = int(round(attention_size*voters_num,0)) # k is the number of people voting based on the attention size
                attention_votes = random.choices(votes, k=k)
                if sum(attention_votes) > k/2: attention_label = 1
                else: attention_label = 0
            else:
                attention_label = None
            attention_label_arr.append(attention_label)
        attention_label_arr_ALL.append(attention_label_arr)
    # print(attention_label_arr_ALL)
    return attention_label_arr_ALL


