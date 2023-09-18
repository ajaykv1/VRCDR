# Vietoris-Rips Complex: A New Direction for Cross-Domain Cold-Start Recommendation

Abstract:

Cross-domain recommendation (CDR) has emerged as a promising solution to alleviating the cold-start problem by leveraging information from an auxiliary source domain to generate recommendations in a target domain. Most CDR techniques fall into a category known as ***bridge-based methods***, but many of them fail to account for the structure and rating behavior of target users from the source domain into the recommendation process. Therefore, we present a novel framework called Vietoris-Rips Complex for Cross-Domain Recommendation framework called (VRCDR), which utilizes the Vietoris-Rips Complex (a technique from computational geometry) to understand the underlying structure in user behavior from the source domain, and include the learned information into  recommendations in the target domain to make the recommendations more personalized to users' niche preferences. Extensive experiments on large, real-world datasets demonstrate that VRCDR consistently improves recommendations compared to state-of-the-art bridge-based CDR methods.  

# Usage

This is the code repository for our research project. This repository contains the files used to replicate the results of VRCDR from the paper. There are 4 main directories:

1. `dataset`: Contains datasets from the Amazon Review dataset that we use in our study. The dataset is quite large, so we did not provide it in this repository, but please download the appropriate files from ***http://jmcauley.ucsd.edu/data/amazon/*** and place them in this folder. 
2. `src`: Contains the main source code files that are used for pre-training models, computing the Vietoris-Rips Complex, and generating results for VRCDR.
3. `project_data`: This is where all the CSV data is stored after being parsed, and it also contains information that the source files use for computation.
4. `stats`: This contains the RMSE and MAE values for each target user based on VRCDR's performance, which is used for the two-tailed paired t-test on our study.

# Running Experiments for VRCDR

In our paper, we have 4 different CDR tasks, and we run experiments using 3 different cold-start ratios (20%, 50%, and 70%). We will show how you can run experiments for **Task 2** using **20%** as the cold-start percentage.

1. Run `save_dataframes_one.py` from the **save_dataframes** folder.
   - To run it, use the command `python .src/save_dataframes/save_dataframes_one.py`
   - This will parse the json files associated with the domains in the **dataset** folder, and store the train/test csv files in project data.

2. Pre-train the source and target domain base models by running the following commands:
   - `python ./src/source_pre_training.py 8_2 music movie`
   - `python ./src/target_pre_training.py 8_2 music movie`
   - The argument **8_2** refers to the cold-start ratio, where 80% is used as training and 20% are test cold-start users
   - The argument **music** refers to the target domain
   - The argument **movie** refers to the source domain
   - These arguments are needed when pre-training, because they are used to locate the train/test splits for the source and target data
  
3. Save content-embeddings for items in the source and target domains using the following commands:
   - `python ./src/content_info_src.py 8_2 music movie`
   - `python ./src/content_info_tgt.py 8_2 music movie`
   - This will save the content embeddings for the source and target domain (retrieved from **DistilBERT**) so that we can use them in VRCDR later
   - Having them stored can assure easy access to embeddings, rather than re-computing them many times
  
4. Create and store the characteristic vectors for users and items using the Vietoris-Rips Complex with the following command:
   - `python ./src/rips_centroid.py 8_2 music movie`
   - This command will save the characteristic vectors for users and items using the Rips complex, so that we can use them in our final model
  
5. Run VRCDR and output the RMSE and MAE values for the model on the test set (extreme cold-start users) with the followinig command:
   - `python ./src/hybrid_bridge.py 8_2 music movie`
   - This will output the RMSE and MAE values for 20% extreme cold-start users for task 2

## Command line arguments for different tasks and different percentage of cold start users

Above, we showed the example of how to run **Task2** with **20%** cold-start test users. Below are the different ratios you can use in command line for cold-start users, and different command line arguments for the tasks. 

### Ratios for cold-start users:

1. `8_2`: 80% used as train set, and 20% are cold-start test users 
2. `5_5`: 50% used as train set, and 50% are cold-start test users
3. `3_7`: 30% used as train set, and 70% are cold-start test users

### Domain combinations for each task:

Task 1: `electronics movie`
Task 2: `music movie`
Task 3: `food movie`
Task 4: `games electronics`

Use the above command line arguments to replicate the results of VRCDR that were presented from in our paper. 

## Reference

For parsing the datasets into train/test splits, we use some of the code provided by the authors of PTUPCDR.
`Zhu, Yongchun, et al. "Personalized Transfer of User Preferences for Cross-domain Recommendation." Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 2022.`








