#!/usr/bin/env python
# coding: utf-8

# In[29]:


from elasticsearch import Elasticsearch
import os

# In[30]:


es = Elasticsearch(
    "https://aws-chatbot.es.us-central1.gcp.cloud.es.io",
    basic_auth=("elastic","YTEJkgvx3NX19njbGyDF40Lo")
   

)
es.ping()


# ### Prepare the data

# In[31]:


import pandas as pd


current_dir = os.path.dirname(__file__)
csv_file_path = os.path.join(current_dir, '..', 'dataset', 'consolidated_data.csv')

# Load data
df = pd.read_csv(csv_file_path).loc[:1500]
df.head()


# #### Check NA values
# 

# In[32]:


df.isna().value_counts()


# In[5]:


df.fillna("None", inplace=True)


# ### Convert the relevant field to Vector using BERT model

# In[33]:


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')


# In[34]:


df["ResponseVector"] = df["response"].apply(lambda x: model.encode(x))


# In[35]:


df.head()


# In[36]:


es.ping()


# ### Create new index in ElasticSearch

# In[37]:


from indexMapping import indexMapping

try:
    es.indices.create(index="all_patterns_v1", mappings=indexMapping) 
except Exception as e:
    pass


# ### Ingest the data into index

# In[38]:


record_list = df.to_dict("records")


# In[39]:


for record in record_list:
    try:
        es.index(index="all_patterns_1500", document=record, id=record["id"])
    except Exception as e:
        print(e)


# In[40]:


es.count(index="all_patterns_1500")


# ### Search the data

# In[41]:


input_keyword = " Billing of Amazon EC2 systems begin and end?"
vector_of_input_keyword = model.encode(input_keyword)

query = {
    "field": "ResponseVector",
    "query_vector": vector_of_input_keyword,
    "k": 1,  # Set k to 1 to get only the top result
    "num_candidates": 1500,
}

res = es.knn_search(index="all_patterns_1500", knn=query, source=["pattern", "response"])
hits = res["hits"]["hits"]


if hits:
    best_match = hits[0]
    print("Best Matching Result:")
    print("Pattern:", best_match["_source"]["pattern"])
    print("Response:", best_match["_source"]["response"])
else:
    print("No matching results found.")


# In[42]:


input_keyword = "Billing of Amazon EC2 systems begin and end?"
vector_of_input_keyword = model.encode(input_keyword)

query = {
    "field" : "ResponseVector",
    "query_vector" : vector_of_input_keyword,
    "k" : 3,
    "num_candidates" : 1500, 
}

res = es.knn_search(index="all_patterns_1500", knn=query , source=["pattern","response"])
res["hits"]["hits"]


# In[43]:


rdf = df.sample(frac=0.01)


# In[44]:


rdf.head()


# In[63]:


def search(input_keyword):
    # model = SentenceTransformer('all-mpnet-base-v2')
    # input_keyword = "Billing of Amazon EC2 systems begin and end?"
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
        "field" : "ResponseVector",
        "query_vector" : vector_of_input_keyword,
        "k" : 3,
        "num_candidates" : 1500, 
    }

    res = es.knn_search(index="all_patterns_1500", knn=query , source=["pattern","response"])
    results = res["hits"]["hits"]

    return results




# In[74]:


arr_of_actual_responses = rdf['response'].tolist()
arr_of_predicted_responses = []

for index, row in rdf.iterrows():
    result = search(row["pattern"])
    print(f"Pattern: {result[1]['_source']['pattern']} ")
    print(f"Response: {result[1]['_source']['response']}")
    arr_of_predicted_responses.append(result[1]['_source']['response'])


# In[75]:


# print(rdf['response'].tolist())
print(arr_of_predicted_responses)


# In[83]:


def precision_for_k(arr_of_actual_responses, arr_of_predicted_responses, k):
    sum = 0

    for i in range(len(arr_of_actual_responses)-1):
        if (arr_of_actual_responses[i] == arr_of_predicted_responses[i]) :
            
            sum += 1
        # else:
            # print(f"{arr_of_actual_responses[i]} !=== {arr_of_predicted_responses[i]}")
    precision = sum / k if k > 0 else 0
    print(f"precision : {precision}")

k = len(rdf.index)
precision_for_k(arr_of_actual_responses, arr_of_predicted_responses, k)
    

