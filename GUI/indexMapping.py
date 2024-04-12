indexMapping = {
    "properties":{
        "id":{
            "type":"long"
        },
        "patterns":{
            "type":"text"
        },
        "PatternVector":{
            "type":"dense_vector",
            "dims": 768,
            "index":True,
            "similarity": "l2_norm"
        }

    }
}