#-------------------------DeepDAO data from pro API - https://api.deepdao.io/pro#/-------------------------

from pathlib import Path
import json
import pandas as pd
import requests
import ast

#------organizations------
# extract the data from json text file
all_organizations = open('all organizations.txt', 'r', encoding='utf-8').read()
all_organizations_dict = json.loads(all_organizations)
# print(all_organizations_dict)

# create a list with all organizations id
list_organizations_id = []
for dict in all_organizations_dict['data']['resources']:
    organization_id = dict['organizationId']
    list_organizations_id.append(organization_id)
# print(len(list_organizations_id))
# print(list_organizations_id)


#------active proposals------
# extract the data from json text file
all_active_proposals = open('all active proposals.txt', 'r', encoding='utf-8').read()
all_active_proposals_dict = json.loads(all_active_proposals)
# print(all_active_proposals_dict)


# ------all proposals (per organization)------
def jsonToDict(jsonString): # from json string to dictionary
    json_data = json.dumps(jsonString)
    aDict = json.loads(json_data)
    return aDict

def dictToDf(a_dict): # from dictionary to data frame
    df = pd.DataFrame()
    df['organizationId'] = [a_dict.get('organizationId', None)]
    df['platform'] = [a_dict.get('platform', None)]
    df['title'] = [a_dict.get('title', None)]
    df['description'] = [a_dict.get('description', None)]
    df['proposer'] = [a_dict.get('proposer', None)]
    df['proposalCreatedAt'] = [a_dict.get('proposalCreatedAt', None)]
    df['proposalNativeId'] = [a_dict.get('proposalNativeId', None)]
    df['endedAt'] = [a_dict.get('endedAt', None)]
    df['status'] = [a_dict.get('status', None)]
    df['totalVotes'] = [a_dict.get('totalVotes', None)]
    df['choicesMappedToScores'] = [a_dict.get('choicesMappedToScores', None)]
    df['proposalUrl'] = [a_dict.get('proposalUrl', None)]
    df['proposerProfilePageUrl'] = [a_dict.get('proposerProfilePageUrl', None)]
    return df

# # extract the data from DeepDAO API
# # in the API you need to insert organization id and it return its proposals -
# # thats the reason why there is no json file and we need to create it.
# headers = {'accept': 'application/json', 'x-api-key': 'Gjdt2eO1vS800t74jlon286QpTfPhphS2iNLL3Fw'} #premission
# all_proposals_dict = {}
# for organization_id in list_organizations_id: # go over all the organizations and for each organization extract its proposals
#     print("-----------------",organization_id)
#     r = requests.get('https://api.deepdao.io/v0.1/proposals/dao_proposals/'+organization_id+'?limit=1000&page=1',headers=headers)
#     # print(r.json())
#     all_proposals_dict[organization_id] = r.json()
# print(all_proposals_dict)
# exDict = {'all_proposals_dict': all_proposals_dict}
# with open('all_proposals_dict.txt', 'w') as file: # save the proposals to jason text file for re-use
#      file.write(json.dumps(exDict)) # use `json.loads` to do the reverse

# # extract the data (all proposals) from json text file
all_proposals = open("all_proposals_dict.txt", 'r', encoding='utf-8').read()
all_proposals_dict = json.loads(all_proposals)
# print(all_proposals_dict['all_proposals_dict'].keys())
jsonString = all_proposals_dict['all_proposals_dict']
aDict = jsonToDict(jsonString)
organizations = aDict.keys()
# print(organizations)
# Arrangement in a convenient dataframe
df_all_proposals = pd.DataFrame()
for organization_id in organizations:
    print(organization_id)
    if aDict[organization_id]:
        for proposal in aDict[organization_id]:
            json_string = proposal
            dict_proposal = jsonToDict(json_string)
            df = dictToDf(dict_proposal)
            df_all_proposals = pd.concat([df_all_proposals, df], ignore_index=True)
            print(df_all_proposals)
df_all_proposals.to_csv('df_all_proposals.csv')

# Examples of organizations with interesting data
df_all_proposals = pd.read_csv('df_all_proposals.csv')
print(df_all_proposals['organizationId'].value_counts())

# PancakeSwap DAO
PancakeSwap_data = df_all_proposals[df_all_proposals['organizationId'] == 'da9956dc-8a87-40c0-a066-b8991d67e574']
PancakeSwap_data.to_csv('PancakeSwap_data.csv')
print(PancakeSwap_data['status'].value_counts())

# Decentraland DAO
Decentraland_data = df_all_proposals[df_all_proposals['organizationId'] == '60c9b31c-4495-4028-aeac-eb7bb117fece']
Decentraland_data.to_csv('Data DeepDAO.csv')

# def get_DeepDAOData():
#     df_DeepDAOData = pd.read_csv('Data DeepDAO.csv')
#     df_DeepDAOData = df_DeepDAOData[(df_DeepDAOData['status'] == 'ACCEPTED') | (df_DeepDAOData['status'] == 'REJECTED')]
#     df_DeepDAOD = df_DeepDAOData.loc[:, ['description', 'status'], ].reset_index().replace({'status': {'ACCEPTED': 1, 'REJECTED': 0}})
#     df_DeepDAOD.rename(columns={'description': 'proposal', 'status': 'label'}, inplace=True)
#     # print(df_DeepDAOD)
#     return(df_DeepDAOD)