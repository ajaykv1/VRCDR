import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
#from transformers import BertTokenizer, TFBertModel
from transformers import DistilBertTokenizer, TFDistilBertModel
import math
#from tensorflow.keras import mixed_precision
import time
import argparse
from tensorflow.keras.callbacks import EarlyStopping
#%%
start_time = time.time()
#mixed_precision.set_global_policy('mixed_float16')
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

# Use map to replace the 'iid' column with the corresponding embeddings from the dictionaries
description_embeddings = np.array(src_data['iid'].map(description_embeddings_dict).tolist())
category_embeddings = np.array(src_data['iid'].map(category_embeddings_dict).tolist())
#%%
print(src_data)
print(src_data.columns)
print(len(src_data))
#%%
n_users = max(src_data['uid'])
n_items = max(src_data['iid'])

embedding_dim = 16

# Define a function to create a MLP model
def create_model(n_users, n_items, embedding_dim, learning_rate):
    # User and item embeddings
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(n_users+1, embedding_dim, name='user_embedding')(user_input)

    item_input = Input(shape=(1,), name='item_input')
    item_embedding = Embedding(n_items+1, embedding_dim, name='item_embedding')(item_input)

    user_vecs = Flatten()(user_embedding)
    item_vecs = Flatten()(item_embedding)

    # Category and description embeddings
    category_input = Input(shape=(768,), name='category_input')  # Adjust the shape according to your embeddings
    description_input = Input(shape=(768,), name='description_input')  # Adjust the shape according to your embeddings

    # Concatenate all vectors
    input_vecs = Concatenate()([user_vecs, item_vecs, category_input, description_input])
    input_vecs = Dropout(0.2)(input_vecs)
    input_vecs = Dense(32, activation='relu')(input_vecs)
    input_vecs = Dropout(0.2)(input_vecs)
    input_vecs = Dense(32, activation='relu')(input_vecs)

    y_pred = Dense(1, activation='linear', dtype='float32')(input_vecs)

    model = Model(inputs=[user_input, item_input, category_input, description_input], outputs=y_pred)
    model.compile(loss=custom_loss, optimizer=Adam(learning_rate=learning_rate))
    
    return model

# Define a custom loss function
def custom_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1, 5)
    return tf.keras.losses.mean_squared_error(y_true, y_pred)
#%%
# Add them to X
X = [src_data['uid'].values, src_data['iid'].values, category_embeddings, description_embeddings]

# Creating k-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
y = src_data['y'].values

learning_rates = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005]  # specify the learning rates to tune
#%%
# Define AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE
# Define batch size
BATCH_SIZE = 256  # Adjust this as per your requirement

# best_rmse = float('inf')
# best_lr = None
# best_model = None

best_rmse = float('inf')
best_loss = float('inf')
best_lr = None
best_model = None
best_mae = float('inf')

for lr in learning_rates:
    print(f'Start training with learning rate {lr}...')
    fold_no = 1
    for train, val in kfold.split(X[0], y):
    
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        
        # Convert the data to TensorFlow Datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(({"user_input": X[0][train],
                                                            "item_input": X[1][train],
                                                            "category_input": X[2][train],
                                                            "description_input": X[3][train]}, y[train]))
        
        val_dataset = tf.data.Dataset.from_tensor_slices(({"user_input": X[0][val],
                                                          "item_input": X[1][val],
                                                          "category_input": X[2][val],
                                                          "description_input": X[3][val]}, y[val]))
        
        #train_dataset = train_dataset.batch(256)
        #val_dataset = val_dataset.batch(256)

        #train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        #val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/best_models/pre_src_best_model_lr_{lr}_fold_{fold_no}.h5', monitor='val_loss', save_best_only=True)
        
        model = create_model(n_users, n_items, embedding_dim, lr)
        
        # Make sure to include the embeddings in the input data
        model.fit(train_dataset,
                 validation_data=val_dataset,
                 epochs=50,  # Set the number of epochs
                 batch_size=256,  # Set your batch size
                 verbose=1,
                 callbacks=[checkpoint, early_stopping])
        

        # Load the best model
        model.load_weights(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/best_models/pre_src_best_model_lr_{lr}_fold_{fold_no}.h5')

        # Predict on validation set
        y_pred = model.predict(val_dataset)
        
        y_pred = np.clip(y_pred, 1, 5)
        
        # Compute RMSE
        rmse = math.sqrt(mean_squared_error(y[val], y_pred))
        mae = mean_absolute_error(y[val], y_pred)
        # Evaluate on validation set
        loss = model.evaluate(val_dataset)
        
        print(f'Validation loss: {loss}')

        # Compute RMSE
        #rmse = math.sqrt(mean_squared_error(y[val], y_pred))
    
        # Update the best RMSE and learning rate if current RMSE is better
        if rmse < best_rmse:
            # best_rmse = rmse
            # best_lr = lr
            # best_model = model
            best_rmse = rmse
            best_mae = mae
            best_loss = loss
            best_lr = lr
            best_model = model

        print(f'Root Mean Squared Error for fold {fold_no}: {rmse}')
        print(f'Mean Average Error for fold {fold_no}: {mae}')
        print(f'Validation Loss for fold {fold_no}: {loss}')

        fold_no += 1

# Print the best RMSE and learning rate after the training
print(f'Best RMSE: {best_rmse}, Best Learning Rate: {best_lr}')
print(f'Best MAE: {best_mae}, Best Learning Rate: {best_lr}')
print(f'Best Validation Loss: {best_loss}, Best Learning Rate: {best_lr}')

best_model.save(f'../project_data/{args.ratio}/tgt_{args.tgt}_src_{args.src}/src_model.h5')
#%%
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time / 60} minutes")

