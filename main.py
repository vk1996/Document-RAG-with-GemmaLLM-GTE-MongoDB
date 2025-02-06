from db_utils import DButils
from ml_utils import MLutils
from chatpipeline import ChatPipeline
from PIL import Image
import os
import re
from matplotlib import pyplot as plt



uri = "mongodb+srv://dbuser1:G-D4tqamxgBLSB-@cluster0.f9wvd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
root_dir="/home/vk/personal/PlanRadar_Assessment_AI_ML_Engineer/assessment-ml-engineer/data/"
descr_fpath=os.path.join(root_dir,"descriptions.json")
img_dir=os.path.join(root_dir,"images")
img_extension=".png"
pattern = r"\d+\.png"
mlutils=MLutils()
dbutils=DButils(uri,mlutils)
load_descriptions_to_mongodb=False


if load_descriptions_to_mongodb and ".json" in descr_fpath:
    dbutils.load_descriptions_to_mongodb(descr_fpath)
    print('Create Vector index for the updated collection in MongoDB before querying for RAG')
    exit()

chatbot = ChatPipeline(mlutils.model,mlutils.tokenizer)
is_started = False


print("\n RAG pipeline ready to answer your queries \n")
while True:
    print('\n')
    query = str(input("You: "))
    if query.lower() in ["exit", "quit"]:
        break
    if not is_started:
        chatbot.clear_chat()
        source_information = dbutils.get_search_result(query)
        chatbot.add_context(source_information)
        is_started = True
        print('\n ########### Context loaded  ########### \n')

    combined_information = f"{query}"
    response = chatbot.chat(combined_information)
    response = response.split(combined_information)[-1]
    print(f"AI: {response}")
    if img_extension in response:
        img_ids = re.findall(pattern,response)
        for img_id in img_ids:
            img = Image.open(os.path.join(img_dir, img_id))
            plt.imshow(img)
            plt.show()
