[![README](https://img.shields.io/badge/README-md-blue.svg)](../README.md)
[![ru](https://img.shields.io/badge/lang-ru-red.svg)](GLOSSARY_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](GLOSSARY.md)

## Glossary

Classes, their methods and fields, that are responsible for interaction with user.

#### Preprocessing pipeline

A set of transformations of the original dataframe returns a numpy array
that contains a transformed dataframe of multimodal data.

#### CVI predictor

Pretrained metaclassifier, which picks 
[target measure](GLOSSARY.md#target-measure) for any dataframe based on it's meta-features.

#### Target measure

Internal measure (or CVI, Cluster Validity Index), that will be used in 
[optimisation pipeline](GLOSSARY.md#optimisation-pipeline) as optimisation function for [Optuna](https://optuna.org/).

#### Optimisation pipeline

Choice of clustering algorithms and their hyperparameters configurations based on 
optimisation history, fitting clustering model on input dataframe,
and estimating it quality. The aim is find clustering algorithm and it's configuration,
so that it maximises [target measure](GLOSSARY.md#target-measure).

#### Optimisation history

Sequence of clustering algorithms choice, their hyperparameters configurations, fitted clustering model 
on input dataframe, it's quality estimation via [target measure](GLOSSARY.md#target-measure) and consumed time budget.

#### HPO backend

Third-party implementations of [Optuna](https://optuna.org/), or others usually based on Bayes optimisation.

#### Deep learning model

Pretrained machine learning models based on neural networks, which can transform image or text data into numeric vector.

#### Dataframe meta

Information about number, dimensions, distance metrics and types of 
modalities in preprocessed dataframe.

#### Search space

Set of clustering algorithms and set of possible values for their hyperparameters, that will be exploited by
optimisation pipeline to find configuration, which maximises [target measure](GLOSSARY.md#target-measure).

#### Log interpretation
- """CVI prediction iteration %N""" - this log displays the iteration number %N for selecting the final measure
- """Picked %measure in %time""" - this log displays the final time %time for selecting the final measure %measure from the following options:
 - BR: Banfield-Raftery metric
 - OS: Relative separability metric
 - MCR: McClain-Rao metric
 - SYM: Symmetric metric
- """MEASURES %time, {%measure: %score}""" - this log of the evaluation by the selected measure %measure with its value %score and the evaluation time %time by the algorithm output by the log below
- """ALGO %time, %status, {%configuration}""" - this log of the selected model, where
 - %time is the running time in seconds
 - %status is the status of training and using the model with the selected hyperparameters
 - %configuration is the selected model with hyperparameters