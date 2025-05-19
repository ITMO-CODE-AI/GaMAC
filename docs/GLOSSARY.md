#### API

Classes, their methods and fields, that are responsible for interaction with user.

#### Preprocessing pipeline

Set of transformations for original ("raw") Spark dataframe, returns **[SparklingDF](../sparkling/data/dataframe.py)**,
which holds transformed dataframe and extra logic for internal needs.

For more information about transformations see [OVERVIEW.md](OVERVIEW.md#preprocessing-pipeline).

#### Preprocessed data

**[Numpy array of preprocessed multimodal data]()**, which is the 
result of [preprocessing pipeline]().

#### CVI predictor

[Pretrained metaclassifier](), which picks 
[target measure]() for any dataframe based on it's meta-features.

#### Target measure

Internal measure (or CVI, Cluster Validity Index), that will be used in 
[optimisation pipeline]() as optimisation function for [Optuna](#Optuna)
and as a part of arm's reward for multi-armed bandit.

#### Optimisation pipeline

Choice of clustering algorithms and their hyperparameters configurations based on 
[optimisation history](#optimisation-history), fitting clustering model on input dataframe,
and estimating it quality. The aim is find clustering algorithm and it's configuration,
so that it maximises [target measure](#target-measure).

To learn more about optimisation pipeline, see [OVERVIEW.md](OVERVIEW.md#optimisation-pipeline).

#### Optimisation history

Sequence of clustering algorithms choice, their hyperparameters configurations, fitted clustering model 
on input dataframe, it's quality estimation via [target measure](#target-measure) and consumed time budget.

#### HPO backend

Third-party realisations of Hyper Parameter Optimisation algorithms, usually based on Bayes optimisation.

#### Deep learning model

Pretrained machine learning models based on neural networks, which can transform image or text data into numeric vector.

#### Dataframe meta

Information about number, dimensions, distance metrics and types of 
modalities in [preprocessed dataframe](#preprocessed-dataframe).

#### Search space

Set of clustering algorithms and set of possible values for their hyperparameters, that will be exploited by
[optimisation pipeline](#optimisation-pipeline) to find configuration, which maximises [target measure](#target-measure).
