import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

indexName = "all_pattern"

try:
    es = Elasticsearch(
    "https://indexdataipynb-2fuqwweqh4tjnqxxx32q77.streamlit.app//",
    basic_auth=("elastic", "YTEJkgvx3NX19njbGyDF40Lo")
   
    )
except ConnectionError as e:
    print("Connection Error:", e)
    
if es.ping():
    print("Succesfully connected to ElasticSearch!!")
else:
    print("Oops!! Can not connect to Elasticsearch!")
# all good up to here



def search(input_keyword):
    model = SentenceTransformer('all-mpnet-base-v2')
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
    "field": "DescriptionVector",
    "query_vector": vector_of_input_keyword,
    "k": 1,  # Set k to 1 to get only the top result
    "num_candidates": 1500,
}

    res = es.knn_search(index="all_patterns_1500", 
                    knn=query, 
                    source=["pattern", "response"])
    results = res["hits"]["hits"]

    return results

def main():
    st.title("Search Q and A")

    # Input: User enters search query
    search_query = st.text_input("Enter your search query")

    # Button: User triggers the search
    if st.button("Search"):
        if search_query:
            # Perform the search and get results
            results = search(search_query)

            # Display search results
            st.subheader("Search Results")
            for result in results:
                with st.container():
                    if '_source' in result:
                        try:
                            st.header(f"{result['_source']['id']}")
                        except Exception as e:
                            print(e)
                        
                        try:
                            st.write(f"Pattern: {result['_source']['response']}")
                        except Exception as e:
                            print(e)
                        st.divider()

                    
if __name__ == "__main__":
    main()
