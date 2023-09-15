#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:22:09 2023

@author: ajaykrishnavajjala
"""

#%%
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
import time
import argparse
from tensorflow.keras.callbacks import EarlyStopping
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
#%%
meta_data = pd.read_csv(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/train_meta.csv', names=['uid', 'iid', 'y', 'category', 'description', 'pos_seq'])
meta_data = meta_data.drop(['category', 'description', 'pos_seq'], axis=1)

meta_data['uid'] = meta_data['uid'].astype(int)
meta_data['iid'] = meta_data['iid'].astype(int)
meta_data['y'] = meta_data['y'].astype(int)

test_data = pd.read_csv(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/test.csv', names=['uid', 'iid', 'y', 'category', 'description', 'pos_seq'])
test_data = test_data.drop(['category', 'description', 'pos_seq'], axis=1)

test_data['uid'] = test_data['uid'].astype(int)
test_data['iid'] = test_data['iid'].astype(int)
test_data['y'] = test_data['y'].astype(int)
#%%
print("Opening and Reading the Centroids List from Meta-Data")

meta_centroid_list = None
meta_desc_cat_list = None

test_centroid_list = None
test_desc_cat_list = None

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
# Define a function to create a Bridge Model
def bridge_model(dropout_rate):
    # User and item embeddings
    user_emb_input = Input(shape=(16,), name='user_emb_input')
    user_centroid_input = Input(shape=(16,), name='user_centroid_input')
    
    description_centroid_input = Input(shape=(768,), name='desc_centroid_input')
    category_centroid_input = Input(shape=(768,), name='cat_centroid_input')

    # Concatenate all vectors
    input_vecs = Concatenate()([user_emb_input, user_centroid_input, description_centroid_input, category_centroid_input])
    
    input_vecs = Dropout(dropout_rate)(input_vecs)
    input_vecs = Dense(32, activation='tanh')(input_vecs)
    input_vecs = Dropout(dropout_rate)(input_vecs)
    input_vecs = Dense(16, activation='sigmoid')(input_vecs)

    model = Model(inputs=[user_emb_input, user_centroid_input, description_centroid_input, category_centroid_input], 
                  outputs=input_vecs)
    return model
#%%
def create_modified_target_model(embedding_dim, learning_rate, dropout_rate):
    user_embedding = Input(shape=(embedding_dim,), name='user_embedding')  # User embedding

    item_embedding = Input(shape=(embedding_dim,), name='item_embedding')  # Item embedding

    # Category and description embeddings
    category_input = Input(shape=(768,), name='category_input')  # Adjust the shape according to your embeddings
    description_input = Input(shape=(768,), name='description_input')  # Adjust the shape according to your embeddings

    # Concatenate all vectors
    input_vecs = Concatenate()([user_embedding, item_embedding, category_input, description_input])
    input_vecs = Dropout(dropout_rate)(input_vecs)
    input_vecs = Dense(32, activation='relu')(input_vecs)
    input_vecs = Dropout(dropout_rate)(input_vecs)
    input_vecs = Dense(32, activation='relu')(input_vecs)

    y_pred = Dense(1, activation='linear', dtype='float32')(input_vecs)

    model = Model(inputs=[user_embedding, item_embedding, category_input, description_input], outputs=y_pred)
    model.compile(loss=custom_loss, optimizer=Adam(learning_rate=learning_rate))  # Adjust learning rate as per your needs
    
    return model

def get_modified_target_model(target_model, learning_rate, dropout_rate):
    # Load the weights from pre-trained model to the new target model
    pre_trained_model = target_model  # replace with your pre-trained model path
    target_model_mod = create_modified_target_model(16, learning_rate, dropout_rate)  # replace with your user embedding dimension
    
   # # # for new_layer, layer in zip(target_model_mod.layers[7:], pre_trained_model.layers[7:]):
    for new_layer, layer in zip(target_model_mod.layers[6:], pre_trained_model.layers[10:]):
        print(f"New layer: {new_layer.name} - Old layer: {layer.name}")
        print(f"New layer weights shape: {new_layer.get_weights()[0].shape if new_layer.get_weights() else 'No weights'}")
        print(f"Old layer weights shape: {layer.get_weights()[0].shape if layer.get_weights() else 'No weights'}")
        #if len(layer.get_weights())!= 0:
        if len(layer.get_weights())== len(new_layer.get_weights()):
            print(f"New layer: {new_layer.name} - Old layer: {layer.name}")
            print(f"New layer weights shape: {new_layer.get_weights()[0].shape if new_layer.get_weights() else 'No weights'}")
            print(f"Old layer weights shape: {layer.get_weights()[0].shape if layer.get_weights() else 'No weights'}")
          
            new_layer.set_weights(layer.get_weights())
    
    # Make sure all the layers in the target_model_mod are non-trainable
    for layer in target_model_mod.layers:
        layer.trainable = False
        
    return target_model_mod

def create_combined_model(target_model, learning_rate, dropout_rate):
    
    target_model_mod = get_modified_target_model(target_model, learning_rate, dropout_rate)
    # Define the inputs to the combined model
    source_user_embs_input = Input(shape=(16,), name='source_user_embs_input')
    desc_centroids_input = Input(shape=(768,), name='desc_centroids_input')
    cat_centroids_input = Input(shape=(768,), name='cat_centroids_input')
    user_centroids_input = Input(shape=(16,), name='user_centroids_input')
    
    target_item_embs_input = Input(shape=(16,), name='target_item_embs_input')
    target_desc_embs_input = Input(shape=(768,), name='target_desc_embs_input')
    target_cat_embs_input = Input(shape=(768,), name='target_cat_embs_input')
    
    print("WE are in combined MODELLLLLL")
    
    bridge = bridge_model(dropout_rate)  # Set your embedding dimensions

    # Use the bridge model to get new user embeddings
    new_user_embs = bridge([source_user_embs_input, user_centroids_input, desc_centroids_input, cat_centroids_input])
    
    # Get the predicted ratings from the target model using the new user embeddings and existing item embeddings
    y_pred = target_model_mod([new_user_embs, target_item_embs_input, target_desc_embs_input, target_cat_embs_input])
    
    # Create the final model that combines the bridge network and the target model
    combined_model = Model(inputs=[source_user_embs_input, desc_centroids_input, cat_centroids_input, user_centroids_input,
                                   target_item_embs_input, target_desc_embs_input, target_cat_embs_input],
                           outputs=y_pred)
    
    # Set all layers in target model to non-trainable
    for layer in target_model_mod.layers:
        layer.trainable = False
    
    combined_model.compile(loss=custom_loss, optimizer=Adam(learning_rate=learning_rate))
    
    return combined_model

#%%
meta_user_centroids = meta_centroid_list[0]
meta_description_centroids = meta_centroid_list[1]
meta_category_centroids = meta_centroid_list[2]

meta_descriptions = meta_desc_cat_list[0]
meta_categories = meta_desc_cat_list[1]

source_meta_user_embs = source_user_embeddings(meta_data['uid'].values).numpy()
target_meta_item_embs = target_item_embeddings(meta_data['iid'].values).numpy()
#%%
print("----- STATS -----")

print(len(meta_user_centroids))
print(len(meta_description_centroids))
print(len(meta_category_centroids))

print(len(meta_descriptions))
print(len(meta_categories))

print(len(source_meta_user_embs))
print(len(target_meta_item_embs))
#%%
X_meta = [
    source_meta_user_embs, target_meta_item_embs,
    meta_descriptions, meta_categories,
    meta_description_centroids, meta_category_centroids, meta_user_centroids
    ]

y_meta = meta_data['y'].values
#%%
y_train, y_val = train_test_split(y_meta, test_size=0.2, random_state=42)

X_train = []
X_val = []

# Split each component in X_meta
for component in X_meta:
    X_train_component, X_val_component = train_test_split(component, test_size=0.2, random_state=42)
    X_train.append(X_train_component)
    X_val.append(X_val_component)
#%%
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#%%
# Define AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE
# Define batch size
BATCH_SIZE = 256  # Adjust this as per your requirement

best_rmse = float('inf')
best_loss = float('inf')
best_lr = None
best_model = None
best_mae = float('inf')

avg_fold_rmse_results  = []
avg_fold_mae_results  = []

learning_rates = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005]
dropout_rates = [0.2]

for dr in dropout_rates:
    for lr in learning_rates:
        print(f'Start training with learning rate {lr}...')
        average_rmse = 0
        average_mae = 0
        fold_no = 1
        for train, val in kfold.split(X_train[0], y_train):
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            
            train_dataset_dict = {
                'source_user_embs_input': X_train[0][train],
                'desc_centroids_input': X_train[4][train],
                'cat_centroids_input': X_train[5][train],
                'user_centroids_input': X_train[6][train],
                'target_item_embs_input': X_train[1][train],
                'target_desc_embs_input': X_train[2][train],
                'target_cat_embs_input': X_train[3][train]
            }
    
            # Repeat the same process for the validation dataset
            val_dataset_dict = {
                'source_user_embs_input': X_train[0][val],
                'desc_centroids_input': X_train[4][val],
                'cat_centroids_input': X_train[5][val],
                'user_centroids_input': X_train[6][val],
                'target_item_embs_input': X_train[1][val],
                'target_desc_embs_input': X_train[2][val],
                'target_cat_embs_input': X_train[3][val]
            }
            
            # Create TensorFlow Datasets for training
            train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset_dict, y_train[train]))
            val_dataset = tf.data.Dataset.from_tensor_slices((val_dataset_dict, y_train[val]))
            
            val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
            train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            checkpoint = ModelCheckpoint(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/best_models/tgt_best_model_lr_{lr}_fold_{fold_no}.h5', monitor='val_loss', save_best_only=True)
            
            # Initialize the bridge model and combined model
            #bridge_model = bridge_model()  # Set your embedding dimensions
            combined_model = create_combined_model(target_model, lr, dr)
            
            print(combined_model.summary())
            
            # Fit the model on the training data
            combined_model.fit(train_dataset,
                                         validation_data=val_dataset,
                                         epochs=50,  # Set the number of epochs
                                         batch_size=256,  # Set your batch size
                                         verbose=1,
                                         callbacks=[checkpoint, early_stopping])
            
            # Load the best model
            combined_model.load_weights(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/best_models/tgt_best_model_lr_{lr}_fold_{fold_no}.h5')
    
            # Predict on validation set
            y_pred = combined_model.predict(val_dataset)
            y_pred = np.clip(y_pred, 1, 5)
    
            # Compute RMSE
            rmse = math.sqrt(mean_squared_error(y_train[val], y_pred))
            mae = mean_absolute_error(y_train[val], y_pred)
            # Evaluate on validation set
            loss = combined_model.evaluate(val_dataset)
            
            print(f'Validation loss: {loss}')
            
            # Update the best RMSE and learning rate if current RMSE is better
            if rmse < best_rmse:
                best_rmse = rmse
                best_mae = mae
                best_loss = loss
                best_lr = lr
    
            print(f'Root Mean Squared Error for fold {fold_no}: {rmse}')
            print(f'Mean Average Error for fold {fold_no}: {mae}')
            print(f'Validation Loss for fold {fold_no}: {loss}')
            
            average_rmse += rmse
            average_mae += mae
            
            fold_no += 1
            
        average_rmse /= 5.0
        average_mae /= 5.0
        avg_fold_rmse_results.append(average_rmse)
        avg_fold_mae_results.append(average_mae)
#%%
# Print the best RMSE and learning rate after the training
print(f'Best RMSE: {best_rmse}, Best Learning Rate: {best_lr}')
print(f'Best MAE: {best_mae}, Best Learning Rate: {best_lr}')
print(f'Best Validation Loss: {best_loss}, Best Learning Rate: {best_lr}')
print("Best Results are: ")
print(avg_fold_mae_results)
print(avg_fold_rmse_results)

index = np.argmin(avg_fold_rmse_results)
best_lr = learning_rates[index]

print(f'We are training it with the best learning rate: {best_lr}')
best_model = create_combined_model(target_model, best_lr, 0.2)

train_dataset_dict = {
    'source_user_embs_input': X_train[0],
    'desc_centroids_input': X_train[4],
    'cat_centroids_input': X_train[5],
    'user_centroids_input': X_train[6],
    'target_item_embs_input': X_train[1],
    'target_desc_embs_input': X_train[2],
    'target_cat_embs_input': X_train[3]
}

val_dataset_dict = {
    'source_user_embs_input': X_val[0],
    'desc_centroids_input': X_val[4],
    'cat_centroids_input': X_val[5],
    'user_centroids_input': X_val[6],
    'target_item_embs_input': X_val[1],
    'target_desc_embs_input': X_val[2],
    'target_cat_embs_input': X_val[3]
}

# Create TensorFlow Datasets for training
train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset_dict, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((val_dataset_dict, y_val))

val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/best_models/tgt_best_model_lr_{lr}_fold_{fold_no}.h5', monitor='val_loss', save_best_only=True)

# Fit the model on the training data
best_model.fit(train_dataset,
                             validation_data=val_dataset,
                             epochs=50,  # Set the number of epochs
                             batch_size=256,  # Set your batch size
                             verbose=1,
                             callbacks=[checkpoint, early_stopping])

# Load the best model
best_model.load_weights(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/best_models/tgt_best_model_lr_{lr}_fold_{fold_no}.h5')
#%%
print(test_data)

test_user_centroids = test_centroid_list[0]
test_description_centroids = test_centroid_list[1]
test_category_centroids = test_centroid_list[2]

test_descriptions = test_desc_cat_list[0]
test_categories = test_desc_cat_list[1]

source_test_user_embs = source_user_embeddings(test_data['uid'].values).numpy()
target_test_item_embs = target_item_embeddings(test_data['iid'].values).numpy()
#%%
print("----- STATS -----")

print(len(test_user_centroids))
print(len(test_description_centroids))
print(len(test_category_centroids))

print(len(test_descriptions))
print(len(test_categories))

print(len(source_test_user_embs))
print(len(target_test_item_embs))
#%%
X_test = [
    source_test_user_embs, target_test_item_embs,
    test_descriptions, test_categories,
    test_description_centroids, test_category_centroids, test_user_centroids
    ]

y_test = test_data['y'].values
#%%
test_dataset_dict = {
    'source_user_embs_input': X_test[0],
    'desc_centroids_input': X_test[4],
    'cat_centroids_input': X_test[5],
    'user_centroids_input': X_test[6],
    'target_item_embs_input': X_test[1],
    'target_desc_embs_input': X_test[2],
    'target_cat_embs_input': X_test[3]
}

# Create TensorFlow Datasets for training
test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset_dict, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)
#%%
best_rmse = float('inf')
best_mae = float('inf')
best_rmse_preds = None
best_mae_preds = None

rmses = 0
maes = 0

for i in range(10):
    # Make predictions
    y_pred = best_model.predict(test_dataset)
    
    # Flatten y_test and y_pred if they have more than 1 dimension
    y_test_flatten = y_test.ravel()
    y_pred_flatten = y_pred.ravel()
    
    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_test_flatten, y_pred_flatten))
    #print("Root Mean Squared Error: ", rmse)
    
    # Calculate MAE
    mae = mean_absolute_error(y_test_flatten, y_pred_flatten)
    #print("Mean Absolute Error: ", mae)
    
    rmses += rmse
    maes += mae
    
    if rmse < best_rmse:
    #if rmse > best_rmse:
        best_rmse = rmse
        best_rmse_preds = y_pred
    
    if mae < best_mae:
    #if mae > best_mae:
        best_mae = mae
        best_mae_preds = y_pred

print("Avergae Test Root Mean Squared Error: ", rmses/10.0)
print("Average Test Mean Absolute Error: ", maes/10.0)

print("Best Test Root Mean Squared Error: ", best_rmse)
print("Best Test Mean Absolute Error: ", best_mae)

#%%
print("len of RMSE Preds = " + str(len(best_rmse_preds)))
print("Length of Ground truth ratings is = " + str(len(y_test)))
user_ids = test_data['uid'].values
print("Length of User IDs is: " + str(len(user_ids)))

user_targets_dict = {}
user_predicts_dict = {}

user_rmse_dict = {}
user_mae_dict = {}
        
# Calculate RMSE for each user
for i,user_id in enumerate(user_ids):
    user_targets = y_test[i]
    user_predicts = best_rmse_preds[i]
    if user_id not in user_targets_dict:
        user_targets_dict[user_id] = []
        user_predicts_dict[user_id] = []
    user_targets_dict[user_id].append(user_targets) 
    user_predicts_dict[user_id].append(user_predicts)

for user_id, user_target in user_targets_dict.items():
    user_predict = user_predicts_dict[user_id]
    
    user_rmse = math.sqrt(mean_squared_error(user_target, user_predict))
    user_rmse_dict[user_id] = user_rmse
    
    user_mae = mean_absolute_error(user_target, user_predict)
    user_mae_dict[user_id] = user_mae
    
print("Length of user RMSE dict is: " + str(len(user_rmse_dict.keys())))
print("User RMSE Dict: " + str(list(user_rmse_dict.items())[0]))

print("Length of user MAE dict is: " + str(len(user_mae_dict.keys())))
print("User MAE Dict: " + str(list(user_mae_dict.items())[0]))

# Create DataFrames
df_rmse = pd.DataFrame(list(user_rmse_dict.items()), columns=['user_id', 'rmse'])
df_mae = pd.DataFrame(list(user_mae_dict.items()), columns=['user_id', 'mae'])

print("RMSE DF Head")
print(df_rmse)

print("MAE DF Head")
print(df_mae)

# Merge DataFrames based on 'user_id'
merged_df = df_rmse.merge(df_mae, on='user_id')

print("MERGED Head")
print(merged_df)  # Print first 5 rows to check

print("Saving Merged Dataframe to CSV")
# Save merged_df as a CSV file
merged_df.to_csv(f'../stats/{args.ratio}/tgt_{args.tgt}_src_{args.src}/{args.ratio}_{args.tgt}_{args.src}_stats.csv', index=False)

#%%
user_ids = test_data['uid'].values
item_ids = test_data['iid'].values
test_descriptions = test_desc_cat_list[0]
test_categories = test_desc_cat_list[1]

y_test = test_data['y'].values

#%%
X_test = [
    user_ids, item_ids,
    test_descriptions, test_categories
    ]

y_test = test_data['y'].values
#%%
test_dataset_dict = {
    'user_input': X_test[0],
    'item_input': X_test[1],
    'category_input': X_test[3],
    'description_input': X_test[2]
}

# Create TensorFlow Datasets for training
test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset_dict, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

y_pred = target_model.predict(test_dataset)

# Flatten y_test and y_pred if they have more than 1 dimension
y_test_flatten = y_test.ravel()
y_pred_flatten = y_pred.ravel()

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(y_test_flatten, y_pred_flatten))
#print("Root Mean Squared Error: ", rmse)

# Calculate MAE
mae = mean_absolute_error(y_test_flatten, y_pred_flatten)

print("Test Root Mean Squared Error: ", rmse)
print("Test Mean Absolute Error: ", mae)
#%%
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time / 60} minutes")


























