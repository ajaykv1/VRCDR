#%%
import ast
import gudhi as gd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from transformers import DistilBertTokenizer, TFDistilBertModel
import math
import time
from scipy.spatial.distance import pdist
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
# Define a custom loss function
def custom_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1, 5)
    return tf.keras.losses.mean_squared_error(y_true, y_pred)
#%%
# Load the pre-trained model
source_model = load_model(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/src_model.h5',
                          custom_objects={'custom_loss': custom_loss})
print(source_model.summary())
# Assuming your model's architecture is such that the layer from which you want to extract embeddings is named 'embedding'
source_user_embeddings = source_model.get_layer('user_embedding')
source_item_embeddings = source_model.get_layer('item_embedding')

# Load the pre-trained model
target_model = load_model(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/tgt_model.h5',
                          custom_objects={'custom_loss': custom_loss})
print(target_model.summary())
# Assuming your model's architecture is such that the layer from which you want to extract embeddings is named 'embedding'
target_user_embeddings = target_model.get_layer('user_embedding')
target_item_embeddings = target_model.get_layer('item_embedding')

# Initialize the tokenizer and the model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
#%%
# Function to compute BERT embeddings in batches
def compute_embeddings(texts):
    inputs = tokenizer(texts.tolist(), return_tensors='tf', truncation=True, padding=True, max_length=128)
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    embeddings = outputs[0][:, 1, :].numpy()
    return embeddings
#%%
def calculate_user_centroid(pos_seq, source_item_embeddings):
    pos_seq = np.array(pos_seq[:50])  # Convert the numpy array to a cupy array
    
    if len(list(pos_seq)) == 0:
        return np.zeros(16)
    if len(pos_seq) == 1:
        return source_item_embeddings(pos_seq).numpy()
    if len(pos_seq) == 2:
        return np.mean(source_item_embeddings(pos_seq).numpy(), axis=0)
    
    points = source_item_embeddings(pos_seq).numpy()
    return calculate_rips(points, 1, 1.0)#compute_average(points)#calculate_rips(points)

def calculate_rips(points, num, edge_length):
    # Initialize RipsComplex with the points, using a max edge length of 1
    rips_complex = gd.RipsComplex(points=points, max_edge_length=edge_length)
    # Compute the simplex tree (this computes the Rips complex)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    # Print the number of simplices in the complex
    print(f'Number of simplices: {simplex_tree.num_simplices()}')
    
    total_area = 0
    areas = []
    centroids = []
    count = 0
    # Print the simplices
    for filtered_value in simplex_tree.get_filtration():
        
        if len(list(filtered_value[0])) == 3:
            triangle_embeddings = [points[i] for i in list(filtered_value[0])]
            centroid_embedding = np.mean(triangle_embeddings, axis=0)
            centroids.append(centroid_embedding)
            
            area = get_area(triangle_embeddings, points)
            total_area += area
            areas.append(area)
            count += 1

    areas_np = np.array(areas)
    centroids_np = np.array(centroids)
    if total_area == 0:
        total_area = 1
    normalized_areas = areas_np / total_area
    result = normalized_areas[:, None] * centroids_np
    final_centroid = np.mean(result, axis=0)
    return final_centroid

def get_area(triangle_embs, points):
    
    emb1 = triangle_embs[0]
    emb2 = triangle_embs[1]
    emb3 = triangle_embs[2]
    # Compute the pairwise distances
    a = np.linalg.norm(emb1 - emb2)
    b = np.linalg.norm(emb1 - emb3)
    c = np.linalg.norm(emb2 - emb3)
    # Compute s, the semi-perimeter
    s = (a + b + c) / 2
    # Now compute the area using Heron's formula
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area

def compute_average(points):
    return np.mean(points, axis=0)

def compute_description_embedding(pos_seq, description_dict):
    pos_seq = np.array(pos_seq[:50])  # Convert the numpy array to a cupy array
    
    if len(list(pos_seq)) == 0:
        return np.zeros(768)
    if len(pos_seq) == 1:
        return description_dict[pos_seq[0]]
    if len(pos_seq) == 2:
        #print("Length of DESC IS 2")
        return description_dict[pos_seq[0]]
        #return np.mean([description_dict[pos_seq[0]], description_dict[pos_seq[1]]], axis=0)
    
    vectors = [description_dict[i] for i in pos_seq if i in description_dict]
    points = np.array(vectors)
    average_vector = calculate_rips(points, 2, 20)#compute_average(points)#calculate_rips(points)
    #compute_average(points)#np.mean(vectors, axis=0)
    return average_vector

def compute_category_embedding(pos_seq, category_dict):
   pos_seq = np.array(pos_seq[:50])  # Convert the numpy array to a cupy array
   
   if len(list(pos_seq)) == 0:
       return np.zeros(768)
   if len(pos_seq) == 1:
       return category_dict[pos_seq[0]]
   if len(pos_seq) == 2:
       #print("Length of CAT IS 2")
       return category_dict[pos_seq[0]]
       #return np.mean([category_dict[pos_seq[0]], category_dict[pos_seq[1]]], axis=0)
   
   vectors = [category_dict[i] for i in pos_seq if i in category_dict]
   points = np.array(vectors)
   average_vector = calculate_rips(points, 3, 20)#compute_average(points)#calculate_rips(points)
   #compute_average(points)#np.mean(vectors, axis=0)
   return average_vector
    
#%%
meta_data = pd.read_csv(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/train_meta.csv', names=['uid', 'iid', 'y', 'category', 'description', 'pos_seq'])
#meta_data = meta_data.drop(['category', 'description'], axis=1)
meta_data['pos_seq'] = meta_data['pos_seq'].apply(ast.literal_eval)

print("Full Meta Data")
print(meta_data)

unique_users = meta_data.drop_duplicates(subset='uid')

print("Full Unique Users")
print(unique_users)

#unique_users = unique_users.iloc[:10]

print("Cut Unique Users")
print(unique_users)

# meta_data = meta_data.iloc[:20]
# print(meta_data)

description_dict = None
category_dict = None

print("Opening and Reading the Description dictionary")

# Loading the dictionary
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/src_description_dict.pickle', 'rb') as handle:
    description_dict = pickle.load(handle)
    
print("Opening and Reading the Category dictionary")

# Loading the dictionary
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/src_category_dict.pickle', 'rb') as handle:
    category_dict = pickle.load(handle)

def process_centroid_chunk(chunk):
    #chunk['pos_seq'] = chunk['pos_seq'].apply(ast.literal_eval)
    return [calculate_user_centroid(x, source_item_embeddings) for x in chunk['pos_seq']]

def process_description_chunk(chunk):
    #chunk['pos_seq'] = chunk['pos_seq'].apply(ast.literal_eval)
    return [compute_description_embedding(x, description_dict) for x in chunk['pos_seq']]

def process_category_chunk(chunk):
    #chunk['pos_seq'] = chunk['pos_seq'].apply(ast.literal_eval)
    return [compute_category_embedding(x, category_dict) for x in chunk['pos_seq']]

#############################################

# Meta-Data Centroid Lists

#############################################

# Split DataFrame into chunks
chunk_size = len(unique_users) // 256 #4
#chunks = np.array_split(unique_users, 4)
chunks = np.array_split(unique_users, chunk_size)

user_results = []
for chunk in chunks:
    processed_chunk = process_centroid_chunk(chunk)
    user_results.append(processed_chunk)
    
# Flatten the list of lists into a single list using numpy
user_centroids = [item.tolist() for sublist in user_results for item in sublist]

description_results = []
for chunk in chunks:
    processed_chunk = process_description_chunk(chunk)
    description_results.append(processed_chunk)

# Flatten the list of lists into a single list using numpy
description_centroids = [item.tolist() for sublist in description_results for item in sublist]

category_results = []
for chunk in chunks:
    processed_chunk = process_category_chunk(chunk)
    category_results.append(processed_chunk)

# Flatten the list of lists into a single list using numpy
category_centroids = [item.tolist() for sublist in description_results for item in sublist]

print("Length of user centroids: " + str(len(user_centroids)))
print("Length of description centroids: " + str(len(description_centroids)))
print("Length of category centroids: " + str(len(category_centroids)))

for i in range(len(user_centroids)):
    if len(user_centroids[i]) != 16:
        user_centroids[i] = user_centroids[i][0]
        
for i in range(len(description_centroids)):
    if len(description_centroids[i]) != 768:
        print("Length of DESC Centroid is " + str(len(description_centroids[i])))
        description_centroids[i] = description_centroids[i][0]
        
for i in range(len(category_centroids)):
    if len(category_centroids[i]) != 768:
        category_centroids[i] = category_centroids[i][0]
        

# Create dictionaries to store the embeddings with 'uid' as keys
user_centroid_embeddings_dict = dict(zip(unique_users['uid'], list(user_centroids)))
description_centroid_embeddings_dict = dict(zip(unique_users['uid'], list(description_centroids)))
category_centroid_embeddings_dict = dict(zip(unique_users['uid'], list(category_centroids)))

print("Finished making Dictionary: " + str(len(user_centroid_embeddings_dict)))

print(meta_data)
print(len(user_centroid_embeddings_dict))
print(len(description_centroid_embeddings_dict))
print(len(category_centroid_embeddings_dict))
print(list(user_centroid_embeddings_dict.items())[:20])

# Use map to replace the 'iid' column with the corresponding embeddings from the dictionaries
user_centroid_embeddings = np.array(meta_data['uid'].map(user_centroid_embeddings_dict).tolist())
description_centroid_embeddings = np.array(meta_data['uid'].map(description_centroid_embeddings_dict).tolist())
category_centroid_embeddings = np.array(meta_data['uid'].map(category_centroid_embeddings_dict).tolist())

print("Information about the user, category, and description embedding:")
print(len(user_centroid_embeddings))
print(len(description_centroid_embeddings))
print(len(category_centroid_embeddings))

centroid_list = [user_centroid_embeddings, description_centroid_embeddings, category_centroid_embeddings]
print(len(centroid_list))

print("Saving List of data")

#Saving Category embeddings
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/rips_data/meta_centroids_list.pickle', 'wb') as handle:
    pickle.dump(centroid_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
#############################################

# Meta-Data Description and Category Lists

#############################################

# Define batch size
batch_size = 256

# Create a new dataframe with unique items
unique_items = meta_data.drop_duplicates(subset='iid')

# Fill NaNs with a placeholder string
unique_items['description'] = unique_items['description'].fillna('')
unique_items['category'] = unique_items['category'].fillna('')

# Replace empty strings with a placeholder string
unique_items['description'] = unique_items['description'].replace({'': 'No description'})
unique_items['category'] = unique_items['category'].replace({'': 'No category'})

# Compute embeddings for 'description' and 'category' in the unique items dataframe
meta_description_embeddings = np.vstack([compute_embeddings(batch) for batch in np.array_split(unique_items['description'], len(unique_items) // batch_size)])

meta_category_embeddings = np.vstack([compute_embeddings(batch) for batch in np.array_split(unique_items['category'], len(unique_items) // batch_size)])

# Create dictionaries to store the embeddings with 'iid' as keys
meta_description_embeddings_dict = dict(zip(unique_items['iid'], list(meta_description_embeddings)))
meta_category_embeddings_dict = dict(zip(unique_items['iid'], list(meta_category_embeddings)))

# Use map to replace the 'iid' column with the corresponding embeddings from the dictionaries
meta_description_embeddings = np.array(meta_data['iid'].map(meta_description_embeddings_dict).tolist())
meta_category_embeddings = np.array(meta_data['iid'].map(meta_category_embeddings_dict).tolist())

print("Stats for meta Metadata")
print(len(meta_data))
print(len(meta_description_embeddings))
print(len(meta_category_embeddings))

meta_list = [meta_description_embeddings, meta_category_embeddings]
print(len(meta_list))

print("Saving List of data")

#Saving Category embeddings
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/rips_data/meta_desc_cat_list.pickle', 'wb') as handle:
    pickle.dump(meta_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%
#############################################

# Test-Data Centroid and Desc, Cat List

#############################################

test_data = pd.read_csv(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/test.csv', names=['uid', 'iid', 'y', 'category', 'description', 'pos_seq'])

test_data['pos_seq'] = test_data['pos_seq'].apply(ast.literal_eval)

print("Full Test Data")
print(test_data)

unique_users = test_data.drop_duplicates(subset='uid')

print("Full Unique Users")
print(unique_users)

#unique_users = unique_users.iloc[:10]

print("Cut Unique Users")
print(unique_users)

# meta_data = meta_data.iloc[:20]
# print(meta_data)

# Split DataFrame into chunks
chunk_size = len(unique_users) // 256 #4
if chunk_size == 0:
    chunk_size = 4
#chunks = np.array_split(unique_users, 4)
chunks = np.array_split(unique_users, chunk_size)

user_results = []
for chunk in chunks:
    processed_chunk = process_centroid_chunk(chunk)
    user_results.append(processed_chunk)
    
# Flatten the list of lists into a single list using numpy
user_centroids = [item.tolist() for sublist in user_results for item in sublist]

description_results = []
for chunk in chunks:
    processed_chunk = process_description_chunk(chunk)
    description_results.append(processed_chunk)

# Flatten the list of lists into a single list using numpy
description_centroids = [item.tolist() for sublist in description_results for item in sublist]

category_results = []
for chunk in chunks:
    processed_chunk = process_category_chunk(chunk)
    category_results.append(processed_chunk)

# Flatten the list of lists into a single list using numpy
category_centroids = [item.tolist() for sublist in description_results for item in sublist]

print("Length of user centroids: " + str(len(user_centroids)))
print("Length of description centroids: " + str(len(description_centroids)))
print("Length of category centroids: " + str(len(category_centroids)))

for i in range(len(user_centroids)):
    if len(user_centroids[i]) != 16:
        user_centroids[i] = user_centroids[i][0]
        
for i in range(len(description_centroids)):
    if len(description_centroids[i]) != 768:
        description_centroids[i] = description_centroids[i][0]
        
for i in range(len(category_centroids)):
    if len(category_centroids[i]) != 768:
        category_centroids[i] = category_centroids[i][0]
        

# Create dictionaries to store the embeddings with 'uid' as keys
user_centroid_embeddings_dict = dict(zip(unique_users['uid'], list(user_centroids)))
description_centroid_embeddings_dict = dict(zip(unique_users['uid'], list(description_centroids)))
category_centroid_embeddings_dict = dict(zip(unique_users['uid'], list(category_centroids)))

print("Finished making Dictionary: " + str(len(user_centroid_embeddings_dict)))

print(test_data)
print(len(user_centroid_embeddings_dict))
print(len(description_centroid_embeddings_dict))
print(len(category_centroid_embeddings_dict))
print(list(user_centroid_embeddings_dict.items())[:20])

# Use map to replace the 'iid' column with the corresponding embeddings from the dictionaries
user_centroid_embeddings = np.array(test_data['uid'].map(user_centroid_embeddings_dict).tolist())
description_centroid_embeddings = np.array(test_data['uid'].map(description_centroid_embeddings_dict).tolist())
category_centroid_embeddings = np.array(test_data['uid'].map(category_centroid_embeddings_dict).tolist())

print("Information about the user, category, and description embedding:")
print(len(user_centroid_embeddings))
print(len(description_centroid_embeddings))
print(len(category_centroid_embeddings))

centroid_list = [user_centroid_embeddings, description_centroid_embeddings, category_centroid_embeddings]
print(len(centroid_list))

print("Saving List of data")

#Saving Category embeddings
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/rips_data/test_centroids_list.pickle', 'wb') as handle:
    pickle.dump(centroid_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
#############################################

# Meta-Data Description and Category Lists

#############################################

# Define batch size
batch_size = 256

# Create a new dataframe with unique items
unique_items = test_data.drop_duplicates(subset='iid')

# Fill NaNs with a placeholder string
unique_items['description'] = unique_items['description'].fillna('')
unique_items['category'] = unique_items['category'].fillna('')

# Replace empty strings with a placeholder string
unique_items['description'] = unique_items['description'].replace({'': 'No description'})
unique_items['category'] = unique_items['category'].replace({'': 'No category'})

# Compute embeddings for 'description' and 'category' in the unique items dataframe
test_description_embeddings = np.vstack([compute_embeddings(batch) for batch in np.array_split(unique_items['description'], len(unique_items) // batch_size)])

test_category_embeddings = np.vstack([compute_embeddings(batch) for batch in np.array_split(unique_items['category'], len(unique_items) // batch_size)])

# Create dictionaries to store the embeddings with 'iid' as keys
test_description_embeddings_dict = dict(zip(unique_items['iid'], list(test_description_embeddings)))
test_category_embeddings_dict = dict(zip(unique_items['iid'], list(test_category_embeddings)))

# Use map to replace the 'iid' column with the corresponding embeddings from the dictionaries
test_description_embeddings = np.array(test_data['iid'].map(test_description_embeddings_dict).tolist())
test_category_embeddings = np.array(test_data['iid'].map(test_category_embeddings_dict).tolist())

print("Stats for Test Metadata")
print(len(test_data))
print(len(test_description_embeddings))
print(len(test_category_embeddings))

test_list = [test_description_embeddings, test_category_embeddings]
print(len(test_list))

print("Saving List of data")

#Saving Category embeddings
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/rips_data/test_desc_cat_list.pickle', 'wb') as handle:
    pickle.dump(test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
print("Opening and Reading the Centroids List from Meta-Data")

# Loading the dictionary
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/rips_data/meta_centroids_list.pickle', 'rb') as handle:
    meta_centroid_list = pickle.load(handle)
    print("meta_centroid_list Info (Len): ")
    print(len(meta_centroid_list))
    
print("Opening and Reading the metadata List from Meta-Data")

# Loading the dictionary
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/rips_data/meta_desc_cat_list.pickle', 'rb') as handle:
    meta_desc_cat_list = pickle.load(handle)
    print("meta_desc_cat_list Info (Len): ")
    print(len(meta_desc_cat_list))
    
print("Opening and Reading the Centroids List from Test-Data")

# Loading the dictionary
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/rips_data/test_centroids_list.pickle', 'rb') as handle:
    test_centroid_list = pickle.load(handle)
    print("test_centroid_list Info (Len): ")
    print(len(test_centroid_list))
    
print("Opening and Reading the metadata List from Meta-Data")

# Loading the dictionary
with open(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/rips_data/test_desc_cat_list.pickle', 'rb') as handle:
    test_desc_cat_list = pickle.load(handle)
    print("test_desc_cat_list Info (Len): ")
    print(len(test_desc_cat_list))
#%%
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time / 60} minutes")


print("USER IS AVERGE, CONTENT IS AVERAGE!!!!")






