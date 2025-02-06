import json
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd



class DButils:

    def __init__(self,mongo_uri,ml_client):

        mongo_client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        db = mongo_client["dataset_rag"]
        self.collection = db["dataset_rag_collections"]
        print("DB loaded")
        self.ml_client=ml_client
        self.df = None


    def load_descriptions_to_mongodb(self,descr_fpath):
        """
            reads the description file
            creates dataframe of descriptions and apply OpenClip text embedding
            pushes the embedding to MongoDB cluster
            Make sure to create MongoDB Atlas search
        """

        descriptions = list(json.load(open(descr_fpath, mode="r")).items())

        lines = []

        for description in descriptions:
            if not description[1].strip():
                continue
            lines.append(f"The image id is {description[0]} and it's description is {description[1]}")

        self.df = pd.DataFrame(lines, columns=['text'])
        self.df = self.df.dropna()
        self.df["embedding"] = self.df["text"].apply(self.get_text_embedding)
        self.collection.delete_many({})
        self.collection.insert_many(self.df.to_dict("records"))
        print("Data ingestion into MongoDB completed")
        #df.head()

    def get_text_embedding(self,text):
        return self.ml_client.get_text_embedding(text)

    def vector_search(self,user_query):
        """
        Perform a vector search in the MongoDB collection based on the user query.
        """

        # Generate embedding for the user query
        query_embedding = self.get_text_embedding(user_query)

        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        # Define the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 150,  # Number of candidate matches to consider
                    "limit": 1,  # Return top k matches
                }
            },
            {
                "$project": {
                    "text": 1,  # Include the plot field
                    "score": {"$meta": "vectorSearchScore"},  # Include the search score
                }
            },
        ]

        # Execute the search
        results = self.collection.aggregate(pipeline)
        return list(results)

    def get_search_result(self,query):

        get_knowledge = self.vector_search(query)

        search_result = ""
        for result in get_knowledge:
            search_result += f"{result.get('text')}\n"

        return search_result

'''
Numpy implementation of cosine similarity used in MongoDB Vector Search
for understanding purpose

import numpy as np
from scipy.spatial.distance import cdist

def cosine_similarity_numpy(a, b):
    a=a/np.linalg.norm(a, axis=1, keepdims=True)
    b=b/np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T)



a= np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b= np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a.shape,b.shape)
cosine_sim=cosine_similarity_numpy(a,b)
print(cosine_sim)
cosine_sim = 1 - cdist(a, b, metric='cosine')
print(cosine_sim)


'''