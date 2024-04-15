import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import os
import pandas as pd


def connect_to_elasticsearch():
    try:
        es = Elasticsearch(
            "https://aws-chatbot.es.us-central1.gcp.cloud.es.io",
            basic_auth=("elastic", "YTEJkgvx3NX19njbGyDF40Lo")
        )
        if es.ping():
            st.success("Successfully connected to Elasticsearch!")
        else:
            st.error("Could not connect to Elasticsearch.")
        return es
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    
def populate_data(es):
    # Check if the index already exists to prevent re-populating data
    if not es.indices.exists(index="all_patterns_v1"):
        current_dir = os.path.dirname(__file__)
        csv_file_path = os.path.join(current_dir, '..', 'dataset', 'consolidated_data.csv')

        # Load data
        df = pd.read_csv(csv_file_path).loc[:1500]
        df.isna().value_counts()
        df.fillna("None", inplace=True)

        model = SentenceTransformer('all-mpnet-base-v2')
        df["ResponseVector"] = df["response"].apply(lambda x: model.encode(x))
    
        from indexMapping import indexMapping
        try:
             es.indices.create(index="all_patterns_v1", mappings=indexMapping) 
        except Exception as e:
          st.error(f"Failed to create index: {e}")
    
        record_list = df.to_dict("records")
        for record in record_list:
            try:
                es.index(index="all_patterns_1500", document=record, id=record["id"])
            except Exception as e:
                st.error(f"Failed to index document: {e}")

        st.success(f"Data populated to Elasticsearch index 'all_patterns_1500'.")



def search(es, input_keyword):
    if es is None:
        return []
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
        vector_of_input_keyword = model.encode(input_keyword)


        # query = {
        #     "query": {
        #         "script_score": {
        #             "query": {"match_all": {}},
        #             "script": {
        #                 "source": "knn_score",
        #                 "lang": "knn",
        #                 "params": {
        #                     "field": "ResponseVector",
        #                     "query_vector": vector_of_input_keyword,
        #                     "k": 3,
        #                     "num_candidates": 1500
        #                 }
        #             }
        #         }
        #     }
        # }

        # res = es.search(index="all_patterns_1500", body=query)
        # results = res["hits"]["hits"]
        # return results

        query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'ResponseVector') + 1.0",
                        "params": {"query_vector": vector_of_input_keyword}
                    }
                }
            }
        }

        res = es.search(index="all_patterns_1500", body=query, size=1)  # Adjust size as needed
        results = res["hits"]["hits"]
        return results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def main():
    st.title("AI-Based Chatbot for AWS")


    es = connect_to_elasticsearch()
    populate_data(es)

    search_query = st.text_input("Enter your search query")

    if st.button("Search"):
        if search_query:
            results = search(es, search_query)

            st.subheader("Search Results")
            for result in results:
                with st.container():
                    try:
                        st.write(f" {result['_source']['response']}")
                    except Exception as e:
                        st.error(f"Error displaying result: {e}")
                    st.divider()

if __name__ == "__main__":
    main()
