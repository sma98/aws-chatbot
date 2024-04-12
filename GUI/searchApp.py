import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

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
    
# def populate_data():
#     current_dir = os.path.dirname(__file__)
#     csv_file_path = os.path.join(current_dir, '..', 'dataset', 'consolidated_data.csv')

# # Load data
#     df = pd.read_csv(csv_file_path).loc[:1500]
#     df.head()




def search(es, input_keyword):
    if es is None:
        return []
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
        vector_of_input_keyword = model.encode(input_keyword)

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
