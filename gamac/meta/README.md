# CVI predictor based on meta-learning

### How to generate data for assessment
1) Find dataset **\<data-name>** with numeric features in csv format, put it inside */data/<data-name>/orig.csv*
2) Run [generator.py](generator.py) specifying **\<data-name>**. It will generate many partitions, but choose only **PARTITIONS_TO_ESTIMATE** candidates for further procedures.
3) This step produces subfolders *data/<data-name>/\<reducer>* with following contents:
   - **gen.csv** - reduced 2D representation of **orig.csv**, obtained by applying **\<reducer>**
   - **partitions.csv** - **\<index>**'th row defines **\<index>**'th partition labels
   - **producers.json** - **\<index>**'th element describes clustering algo, which produces **\<index>**'th partition
   - **images/\<index>.png** - scatter plot of partition **\<index>**

### How to launch GUI assessment
1) Download and unzip **data.zip** or build your own dataset following guide above.
2) After you obtain dataset, you can launch [assessment gui](gui.py) to compare partitions.
   - Install [tkinter](https://docs.python.org/3/library/tkinter.html) module
   - Run ```python gui.py <ACCESSOR_ID>```, where **<ACCESSOR_ID>** is unique string identifier for accessor
3) To estimate single 2D representation partitions, assessment flow goes the following way:
   - There are always two alternative (**PUSH** and **PULL**) scatter plots to compare
   - Alternative **PULL** is one of the already estimated images. All **PULL** images are already sorted by accessor preferences.
   - Alternative **PUSH** is an unseen previously image, that should be compared with **PULLED** images and hence inserted into according to accessor preferences.
   - To switch between alternatives, click **\<TAB>**.
   - To mark current rendered alternative as better, press **\<SPACE>**.
   - If you mark **PULL** alternative as better, the next **PULLED** image will be compared with present **PUSH** alternative.
   - If you mark **PUSH** alternative as better, it will be placed into **PULLED** image list and the next unseen image become **PUSH** alternative.
   - Comparisons repeat until all **15** images will be inserted into sorted **PULLED** images list.
   - After that you will see red frame, which means result is persisting and the next data assessment is loading.
4) This step produces *data/\<data-name>/\<reducer>/accessors/\<ACCESSOR_ID>.json*, containing markup

### How to prepare data for meta-classifier / meta-regressor build
1) Run [features.py](features.py) to compute meta-features for available datasets. 
This step will produce *data/\<data-name>/\<reducer>/features.txt*, containing serialised list of floats
2) Run [measures.py](measures.py) to estimate partitions by internal measures for available datasets.
This step will produce *data/\<data-name>/\<reducer>/measures.json*, containing monotonically increasing variants of CVI's values.

### How to build meta-classifier / meta-regressor
1) Run [orderings.py](orderings.py) to collect both measures and accessors orderings for markuped data.
This step will produce *common/orderings.json*, representing arg-sorted array for partitions
2) Run [scorings.py](scorings.py) to transform orderings into scalar values for each measure.
This step will produce *common/scorings.json*, representing scalar scores for measures on particular dataset (the lower value - the better correlation between measure and markup)
3) Run [premeta.py](premeta.py) to select top of internal measures, that have higher correlation with accessor markups.
This step will produce *common/pre-meta.json*, containing list of top measures and their scores for datasets
4) Optional. Run [tuning.py](tuning.py) to search for optimal configuration for meta-classifier / meta-regressor
This step can be used to find the top-performant hyperparameters w.r.t F1-score (or SMAPE fpr meta-regressor)
5) Run [build.py](build.py) to get fitted meta-classifier / meta-regressor.
This step will produce pickle files:
   - *classifier.csv* - meta-dataset, that was produced and used to fit meta-classifier
   - *common/classifier-extractor.pkl* - feature selector for meta-classifier
   - *common/classifier-model.pkl* - meta-classifier itself
   - *regressor.csv* - meta-dataset, that was produced and used to fit meta-regressor
   - *common/regressor-extractor.pkl* - feature selector for meta-regressor
   - *common/regressor-model.pkl* - meta-regressor itself
