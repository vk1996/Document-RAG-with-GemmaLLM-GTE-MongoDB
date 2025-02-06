## Document / Dataset Q&A  RAG with Gemma, MongoDB ##

### The notebook showcases following ###
### * OpenClip to create text embeddings of dataset or documents ###
### * Using MongoDB to store the embeddings and run vector index ###
### * Find matches to input query among the embeddings with cosine similarity vector search ###
### * Use Google Gemma 2B LLM to summarise results as text / image based on input query related to dataset or document information ###
### * Huggingface chat pipeline to remember your queries & chat ###

#### sample dataset for inference [download link](https://drive.google.com/file/d/1czJDgLRcGeJY6c162f0SNZTIafoHzrcw/view?usp=drive_link) ####
```commandline
$pip install -r requirements.txt
$python3 main.py
```

![output.png](output.png)

### ToDo ###
### Deploy on AWS Sagemaker ###