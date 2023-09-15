# Vietoris-Rips Complex: A New Direction for Cross-Domain Cold-Start Recommendation

Abstract:

Cross-domain recommendation (CDR) has emerged as a promising solution to alleviating the cold-start problem by leveraging information from an auxiliary source domain to improve recommendations in a target domain. Most CDR techniques fall under a category known as bridge-based methods, but many of them fail to include the structure and rating behavior of target users from the source domain into the recommendation process. Therefore, we propose a novel Vietoris-Rips Complex for Cross-Domain Recommen- dation framework called VRCDR. Specifically, we utilize the Vietoris-Rips Complex (a technique from computational ge- ometry) to understand the underlying structure in user be- haviour from the source domain, and include the learned information into the recommendation process to make rec- ommendations more personalized to users’ niche preferences. Extensive experiments on large real-world datasets show that VRCDR improves recommendations compared to state- of-the-art bridge-based CDR methods.

# Usage

This is the code repository for our research project. This repository contains the files used to replicate the results of VRCDR from the paper. There are 4 main directories:

1. `dataset`: Contains datasets from the Amazon Review dataset that we use in our study. 
2. `src`: Contains the main source code files that are used for pre-training models, computing the Vietoris-Rips Complex, and generating results for VRCDR.
3. `project_data`: This is where all the CSV data is stored after being parsed, and it also contains information that the source files use for computation.
4. `stats`: This contains the RMSE and MAE values for each target user based on VRCDR's performance, which is used for the two-tailed paired t-test on our study.


