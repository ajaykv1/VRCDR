#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:54:31 2023

@author: ajaykrishnavajjala
"""

#%%
import pandas as pd
import numpy as np
import tensorflow as tf
#from transformers import BertTokenizer, TFBertModel
from transformers import DistilBertTokenizer, TFDistilBertModel
import math
import time
import argparse
import pickle
#%%
start_time = time.time()
#%%
# Create the parser
parser = argparse.ArgumentParser(description='Process some command line arguments.')

# Add the arguments
parser.add_argument('ratio', type=str, help='The ratio argument')
parser.add_argument('tgt', type=str, help='The tgt argument')
parser.add_argument('src', type=str, help='The src argument')

# Parse the arguments
args = parser.parse_args()
#%%
src_data = pd.read_csv(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/train_src.csv', names=['uid', 'iid', 'y', 'category', 'description'])
print(src_data)

# Initialize the tokenizer and the model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Define batch size
batch_size =256

# Function to compute BERT embeddings in batches
def compute_embeddings(texts):
    inputs = tokenizer(texts.tolist(), return_tensors='tf', truncation=True, padding=True, max_length=128)
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    embeddings = outputs[0][:, 1, :].numpy()
    return embeddings

# Create a new dataframe with unique items
unique_items = src_data.drop_duplicates(subset='iid')

# Fill NaNs with a placeholder string
unique_items['description'] = unique_items['description'].fillna('')
unique_items['category'] = unique_items['category'].fillna('')

# Replace empty strings with a placeholder string
unique_items['description'] = unique_items['description'].replace({'': 'No description'})
unique_items['category'] = unique_items['category'].replace({'': 'No category'})

# Compute embeddings for 'description' and 'category' in the unique items dataframe
description_embeddings = np.vstack([compute_embeddings(batch) for batch in np.array_split(unique_items['description'], len(unique_items) // batch_size)])

category_embeddings = np.vstack([compute_embeddings(batch) for batch in np.array_split(unique_items['category'], len(unique_items) // batch_size)])

# Create dictionaries to store the embeddings with 'iid' as keys
description_embeddings_dict = dict(zip(unique_items['iid'], list(description_embeddings)))
category_embeddings_dict = dict(zip(unique_items['iid'], list(category_embeddings)))

print("Description Embeddings Info: ")
print(len(description_embeddings_dict))
print(list(description_embeddings_dict.keys())[:5])

print("Category Embeddings Info: ")
print(len(category_embeddings_dict))
print(list(category_embeddings_dict.keys())[:5])

# Use map to replace the 'iid' column with the corresponding embeddings from the dictionaries
description_embeddings = np.array(src_data['iid'].map(description_embeddings_dict).tolist())
category_embeddings = np.array(src_data['iid'].map(category_embeddings_dict).tolist())
#%%
#Saving Description embeddings
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/src_description_dict.pickle', 'wb') as handle:
    pickle.dump(description_embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#Saving Category embeddings
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/src_category_dict.pickle', 'wb') as handle:
    pickle.dump(category_embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%
print("Opening and Reading the Description dictionary")

# Loading the dictionary
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/src_description_dict.pickle', 'rb') as handle:
    description_dict = pickle.load(handle)
    print("Description Embeddings Info: ")
    print(len(description_dict))
    print(list(description_dict.keys())[:5])
    
print("Opening and Reading the Category dictionary")

# Loading the dictionary
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/src_category_dict.pickle', 'rb') as handle:
    category_dict = pickle.load(handle)
    print("Category Embeddings Info: ")
    print(len(category_dict))
    print(list(category_dict.keys())[:5])
#%%
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time / 60} minutes")

























