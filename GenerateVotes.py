import numpy as np
import random


# ----Generate Votes----
def get_votes(labels_test, voters):
    arr_votes = []
    for label in labels_test:
        # print(label)
        rnd = round(random.uniform(0.5, 1),1)
        if label ==0:
            zero = int(voters * rnd)
            one = voters - zero
            # print("label: ",0)
            # print("zero", zero)
            # print("one", one)
        else:
            one = int(voters * rnd)
            zero = voters - one
            # print("label: ", 1)
            # print("zero",zero)
            # print("one", one)
        zeros = (np.zeros(zero)).astype(int)
        ones = (np.ones(one)).astype(int)
        votes = np.concatenate((zeros, ones))
        random.shuffle(votes)
        # print('votes\n',votes)
        arr_votes.append(votes)
    # print(arr_votes)
    return arr_votes

# ----Voting based on attention size----
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


