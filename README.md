# How to use this RAG AI teaching assitance on your own data
## Step 1 - Collect your videos 
Move all your video files to the video folder

## Step 2 - Convert video to mp3
Convert all the video files to mp3 by running video_to_mp3

## Step 3 - Create mp3 to JSON's 
Convert all the mp3 files to json by running mp3_to_json

## Step 4 - Convert JSON files to vector
Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 - Prompt genration and feeding to LLM

Read the joblib file and load it into the memory. Then create a relevent prompt as per the user query and feed it to the LLM
