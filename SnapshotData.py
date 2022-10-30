#-------------------------Snapshot data from GraphQL API - https://hub.snapshot.org/graphql-------------------------

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Functions
def run_query(query):  # A simple function to use requests.post to make the API call. Note the json = section.
    request = requests.post('https://hub.snapshot.org/graphql', json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

def get_query_Proposals(id):
    query_Proposals = """
    query Proposals {
      proposals(first: 10000, skip: 10, where: {space: """ + '"' + id + '"' + """, state: "closed"}, orderBy: "created", orderDirection: desc) {
            id
            title
            body
            choices
            scores
          }
        }
        """
    result_Proposals = run_query(query_Proposals)  # Execute the query
    return (result_Proposals)

def check_Consensus(spaces_id_arr):
    df_DAOs = pd.DataFrame(columns=['DAO_name', 'proposals_amount', 'pass_over90%', 'pass_over70%', 'pass_over50%'])
    DAO_name_arr = []
    proposals_amount_arr = []
    pass_over90_arr = []
    pass_over70_arr = []
    pass_over50_arr = []
    for id in spaces_id_arr:
        result_Proposals = get_query_Proposals(id)  # Execute the query
        # results.append(result_Proposals)
        pass_prec_arr = []
        if len(result_Proposals["data"]["proposals"])>0 and sum(1 for item in result_Proposals["data"]["proposals"] if len(item['choices']) == 2)>0:
            DAO_name_arr.append(id)
            proposals_amount = sum(1 for item in result_Proposals["data"]["proposals"] if len(item['choices']) == 2)
            proposals_amount_arr.append(proposals_amount)
            for proposal in result_Proposals["data"]["proposals"]:
                if proposal and len(proposal['choices']) == 2:
                    sum_ = sum(proposal["scores"])
                    pass_prec = -1 if sum_ == 0 else (proposal["scores"][0]/sum_)*100
                    pass_prec_arr.append(pass_prec)
            pass_over90 = sum(1 for p in pass_prec_arr if p >= 90)
            pass_over90_arr.append([pass_over90, round(pass_over90/proposals_amount*100,2)])
            pass_over70 = sum(1 for p in pass_prec_arr if (p < 90 and p >= 70))
            pass_over70_arr.append([pass_over70, round(pass_over70/proposals_amount*100,2)])
            pass_over50 = sum(1 for p in pass_prec_arr if (p < 70 and p >= 50))
            pass_over50_arr.append([pass_over50, round(pass_over50/proposals_amount*100,2)])

    df_DAOs['DAO_name'] = DAO_name_arr
    df_DAOs['proposals_amount'] = proposals_amount_arr
    df_DAOs['pass_over90%'] = pass_over90_arr
    df_DAOs['pass_over70%'] = pass_over70_arr
    df_DAOs['pass_over50%'] = pass_over50_arr
    return(df_DAOs)

def data_for_NLP(result_Proposals):
    df = pd.DataFrame(columns=['proposal_id', 'proposal_title', 'proposal_body', 'proposal_choices', 'proposal_scores'])
    proposal_id_arr = []
    proposal_title_arr = []
    proposal_body_arr = []
    proposal_choices_arr = []
    proposal_scores_arr = []
    # result_Proposals = get_query_Proposals(id)  # Execute the query
    if len(result_Proposals["data"]["proposals"]) > 0 and sum(1 for item in result_Proposals["data"]["proposals"] if len(item['choices']) == 2) > 0:
        for proposal in result_Proposals["data"]["proposals"]:
            # if proposal and len(proposal['title']) > 0 and len(proposal['choices']) == 2 and sum(proposal['scores'])>0:
            if proposal and len(proposal['title']) > 0 and len(proposal['choices']) == 2 and len(proposal['scores']) == 2:
                proposal_id_arr.append(proposal["id"])
                proposal_title_arr.append(proposal["title"])
                proposal_body_arr.append(proposal["body"])
                proposal_choices_arr.append(proposal["choices"])
                proposal_scores_arr.append(proposal["scores"])
        df['proposal_id'] = proposal_id_arr
        df['proposal_title'] = proposal_title_arr
        df['proposal_body'] = proposal_body_arr
        df['proposal_choices'] = proposal_choices_arr
        df['proposal_scores'] = proposal_scores_arr
    return(df)

# The GraphQL Spaces query
def get_query_Spaces():
    query_Spaces = """
    query Spaces {
      spaces(first: 10000, orderBy: "id") {
        id
        name
        about
        proposalsCount
      }
    }
    """
    return (query_Spaces)

# Data handling
def add_labels(df):
    proposal_label_arr = []
    # print(len(df.index))
    # df.to_csv('yam.csv')
    for index, row in df.iterrows():
        # print(index)
        if row['proposal_scores'][0] > row['proposal_scores'][1]:
            proposal_label_arr.append(1)
        else:
            proposal_label_arr.append(0)
    df['label'] = proposal_label_arr
    return df

def delete_rows(df):
    df1 = pd.DataFrame(columns=['proposal_id', 'proposal_title', 'proposal_body', 'proposal_choices', 'proposal_scores'])
    for index, row in df.iterrows():
        # print(index)
        if sum(row['proposal_scores'])>0:
            df1 = df1.append(row, ignore_index = True)
    return df1

# Examining the data
def invest_in_pass(df):
    sum_all = 0
    sum_pass = 0
    for index, row in df.iterrows():
        sum_all = sum_all + sum(row['proposal_scores'])
        sum_pass = sum_pass + row['proposal_scores'][0]
    # print(sum_pass)
    # print(sum_all)
    if sum_all == 0 :
        investInPass = 0
    else:
        investInPass = round((sum_pass/sum_all)*100,3)
    return (investInPass)

def consensus(df):
    proposals_amount = len(df.index)
    pass_prec_arr = []
    for index, row in df.iterrows():
        sum_ = sum(row['proposal_scores'])
        pass_prec = -1 if sum_ == 0 else (row['proposal_scores'][0] / sum_) * 100
        pass_prec_arr.append(pass_prec)
    pass_over90 = round((sum(1 for p in pass_prec_arr if p >= 90)/proposals_amount)*100,3)
    pass_over70 = round((sum(1 for p in pass_prec_arr if (p < 90 and p >= 70))/ proposals_amount)*100,3)
    return pass_over90

# Execute the query
def result_Spaces():
    result_Spaces = run_query(get_query_Spaces())
    spaces_id_arr = []
    for i in range (len(result_Spaces["data"]["spaces"])):
        if result_Spaces["data"]["spaces"][i]["proposalsCount"]>20:
            spaces_id_arr.append(result_Spaces["data"]["spaces"][i]["id"])
    # print("-----Spaces-----")
    print(spaces_id_arr)
    # print("amount of spaces: ", len(spaces_id_arr))
    return spaces_id_arr

def get_df_basedDAO(DAOname):
    result_Proposals = get_query_Proposals(DAOname)
    df = data_for_NLP(result_Proposals)
    df = delete_rows(df)
    df = add_labels(df)
    return df

#-----------------MAIN--------------------
# # extract the data from graphQL
# headers = {"Accept": "application/jason"} #premission
#
# # 5 interesting DAOs
# df_Snapshot1 = get_df_basedDAO('balancer.eth')
# df_Snapshot2 = get_df_basedDAO('yam.eth')
# df_Snapshot3 = get_df_basedDAO('testbsw.eth')
# df_Snapshot4 = get_df_basedDAO('aave.eth')
# df_Snapshot5 = get_df_basedDAO('aavegotchi.eth')
#
# # Connecting the data
# df_Snapshot = pd.concat([df_Snapshot1,df_Snapshot2,df_Snapshot3,df_Snapshot4,df_Snapshot5])
# df_Snapshot.to_csv('Data Snapshot.csv') # Save

def get_SnapshotData():
    # Read the data
    df_SnapshotData = pd.read_csv('Data Snapshot.csv')
    df_Snapshot = df_SnapshotData.loc[:, ['proposal_body', 'label']]
    df_Snapshot.rename(columns = {'proposal_body':'proposal'}, inplace = True)
    return df_Snapshot