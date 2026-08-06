import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

def create_embedding(text_list):
    r= requests.post('http://localhost:11434/api/embed', json= {
        'model': "bge-m3",
        'input': text_list
    })
    embedding= r.json()['embeddings']
    return embedding

def inference(prompt):
    r= requests.post('http://localhost:11434/api/generate', json= {
        'model': 'llama3.2',
        'prompt': prompt,
        'stream': False

    })

    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question: ")# Taking question from the user
print("Thinking....")
question_to_embedding = create_embedding([incoming_query])[0]

# Find similarity of question embedding with other embedding

similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_to_embedding]).flatten()
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
new_df= df.loc[max_indx]


prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks contaning video title,video number, start time in second, end time in seconds, the text at that time:

{new_df[['title','number','start','end','text']].to_json(orient="records")}
--------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format it just for you) where and how much content is taught in which video (in which video and and what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him you can only answers related to the course
'''
with open('prompt.txt','w') as f:
  f.write(prompt)

response_dict = inference(prompt)
print (response_dict)

with open ("responce.txt", "w") as f:
  f.write(response_dict['response'])
