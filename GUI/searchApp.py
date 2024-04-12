import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import os


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
    
def populate_data():
    current_dir = os.path.dirname(__file__)
    csv_file_path = os.path.join(current_dir, '..', 'dataset', 'consolidated_data.csv')

# Load data
    df = pd.read_csv(csv_file_path).loc[:1500]
    df.head()
    df.isna().value_counts()
    
    df.fillna("None", inplace=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')

    df["ResponseVector"] = df["response"].apply(lambda x: model.encode(x))
    
    from indexMapping import indexMapping
    try:
         es.indices.create(index="all_patterns_v1", mappings=indexMapping) 

    except Exception as e:
      pass
    
    record_list = df.to_dict("records")

    for record in record_list:
     try:
        es.index(index="all_patterns_1500", document=record, id=record["id"])
     except Exception as e:
        print(e)

    es.count(index="all_patterns_1500")    
          




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
            "field": "ResponseVector",
            "query_vector": vector_of_input_keyword,
            "k": 1,  # Set k to 1 to get only the top result
            "num_candidates": 1500,
        }

        res = es.knn_search(index="all_patterns_1500", 
                            knn=query, 
                            source=["pattern", "response"])
        results = res["hits"]["hits"]
        return results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def main():
    st.title("Search Q and A")
    populate_data()

    es = connect_to_elasticsearch()

    search_query = st.text_input("Enter your search query")

    if st.button("Search"):
        if search_query:
            results = search(es, search_query)

            st.subheader("Search Results")
            for result in results:
                with st.container():
                    try:
                        st.header(f"ID: {result['_source']['id']}")
                        st.write(f"Pattern: {result['_source']['response']}")
                    except Exception as e:
                        st.error(f"Error displaying result: {e}")
                    st.divider()

if __name__ == "__main__":
    main()
