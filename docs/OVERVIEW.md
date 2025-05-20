## Overview

### Preprocessing pipeline

*GaMAC* introduces its own dataframe preprocessor, which transforms 
raw dataframe into suitable for further clustering multimodal dataframe. 

Here is the top-level blueprint:

In fact, pipeline processor works with tables, images and texts. 
For text preprocessing following stages are included
* **Categorical features** transforms categorical features with OneHot or Label encoding
* **NaN columns** handles NaN with filling own mean or mode values or dropping features from specific NaN threshold
* **Scaling** recomputes number columns with a new number range
* **Finalize** with transforming whole data into a numpy array

For images and texts:
* **Clip-based preprocessing** transforms images and texts to model embeddings of [openai/clip-vit-large-patch14](https://huggingface.co/sergeyzh/rubert-tiny-turbo), which concatanates with table data

Future work: develop an embedder customisation for users.

### Meta-feature selection pipeline

At this stage, the quality measure classifier is used for the clustering task. 
This classifier allows you to select the optimal measure of quality for a specific dataset. 
At the output of the pipeline, a specific measure is given, which will be used in the search for the best optimization algorithm.

### Optimisation pipeline

This pipeline uses 6 clustering algorithms adapted for GPU acceleration. 
These algorithms are iterated over with different variations of hyperparameters, which uses the Optuna framework.
The pipeline outputs: the best configuration of the clustering algorithm with optimal hyperparameters, the best value of the quality measure, and the history of searching for the configuration of the optimal algorithm.





