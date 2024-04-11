import streamlit as st
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ElasticsearchException
from sentence_transformers import SentenceTransformer


indexName = "all_pattern"

# Initialize Elasticsearch client
try:
    es = Elasticsearch(
        "https://indexdataipynb-2fuqwweqh4tjnqxxx32q77.streamlit.app//",
        basic_auth=("elastic", "YTEJkgvx3NX19njbGyDF40Lo")
    )
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# Ping Elasticsearch to check connection
try:
    if es.ping():
        st.success("Successfully connected to Elasticsearch!")
    else:
        st.error("Could not connect to Elasticsearch.")
        st.stop()
except ElasticsearchException as e:
    st.error(f"Elasticsearch connection failed: {e}")
    st.stop()

# Define the search function
def search(input_keyword):
    model = SentenceTransformer('all-mpnet-base-v2')
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
        "field": "DescriptionVector",
        "query_vector": vector_of_input_keyword,
        "k": 1,  # Set k to 1 to get only the top result
        "num_candidates": 1500,
    }

    try:
        res = es.knn_search(index="all_patterns_1500", knn=query, source=["pattern", "response"])
        return res["hits"]["hits"]
    except ElasticsearchException as e:
        st.error(f"Search query failed: {e}")
        return None

# Define the main function for the Streamlit app
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
            if results:
                st.subheader("Search Results")
                for result in results:
                    with st.container():
                        try:
                            st.header(f"{result['_source']['id']}")
                            st.write(f"Pattern: {result['_source']['response']}")
                        except KeyError as e:
                            st.error(f"Error displaying results: {e}")
            else:
                st.write("No results found.")

if __name__ == "__main__":
    main()
