[![ru](https://img.shields.io/badge/lang-ru-red.svg)](GLOSSARY_RU.md.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](GLOSSARY.md)

## Glossary

Classes, their methods and fields, that are responsible for interaction with user.

#### Preprocessing pipeline

A set of transformations of the original dataframe returns a numpy array
that contains a transformed dataframe of multimodal data.

#### CVI predictor

[Pretrained metaclassifier](), which picks 
[target measure]() for any dataframe based on it's meta-features.

#### Target measure

Internal measure (or CVI, Cluster Validity Index), that will be used in 
[optimisation pipeline]() as optimisation function for [Optuna]().

#### Optimisation pipeline

Choice of clustering algorithms and their hyperparameters configurations based on 
[optimisation history](), fitting clustering model on input dataframe,
and estimating it quality. The aim is find clustering algorithm and it's configuration,
so that it maximises [target measure]().

#### Optimisation history

Sequence of clustering algorithms choice, their hyperparameters configurations, fitted clustering model 
on input dataframe, it's quality estimation via [target measure]() and consumed time budget.

#### Optuna backend

Third-party realisations of Optuna, usually based on Bayes optimisation.

#### Deep learning model

Pretrained machine learning models based on neural networks, which can transform image or text data into numeric vector.

#### Dataframe meta

Information about number, dimensions, distance metrics and types of 
modalities in [preprocessed dataframe]().

#### Search space

Set of clustering algorithms and set of possible values for their hyperparameters, that will be exploited by
[optimisation pipeline]() to find configuration, which maximises [target measure]().
