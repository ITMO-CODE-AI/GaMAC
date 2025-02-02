# CVI predictor based on meta-learning

### How to generate data for assessment
1) Find dataset **\<data-name>** with numeric features in csv format, put it inside */data/<data-name>/orig.csv*
2) Run [generator.py](generator.py) specifying **\<data-name>**. It will generate many partitions, but choose only **PARTITIONS_TO_ESTIMATE** candidates for further procedures.
3) This step produces subfolders *data/<data-name>/\<reducer>* with following contents:
   - **gen.csv** - reduced 2D representation of **orig.csv**, obtained by applying **\<reducer>**
   - **img-\<index>.png** - scatter plot of partition **\<index>**
   - **partitions.csv** - **\<index>**'th row defines **\<index>**'th partition labels
   - **producers.json** - **\<index>**'th element describes clustering algo, which produces **\<index>**'th partition

### How to launch assessment
1) Download and unzip **data.zip** or build your own dataset following guide above.
2) After you obtain dataset, you can launch [assessment gui](gui.py) to compare partitions. 
3) Make sure, to assign your unique **ACCESSOR_IDX** before start (should be non-negative integer).
4) To estimate single 2D representation partitions, assessment flow goes the following way:
    - There are always two alternative (**PUSH** and **PULL**) scatter plots to compare
    - Alternative **PULL** is one of the already estimated images. All **PULL** images are already sorted by accessor preferences.
    - Alternative **PUSH** is an unseen previously image, that should be compared with **PULLED** images and hence inserted into according to accessor preferences.
    - To switch between alternatives, click **\<TAB>**.
    - To mark current rendered alternative as better, press **\<SPACE>**.
    - If you mark **PULL** alternative as better, the next **PULLED** image will be compared with present **PUSH** alternative.
    - If you mark **PUSH** alternative as better, it will be placed into **PULLED** image list and the next unseen image become **PUSH** alternative.
    - Comparisons repeat until all **15** images will be inserted into sorted **PULLED** images list.
    - After that you will see red frame, which means result is persisting and the next data assessment is loading.
5) This step produces *data/\<data-name>/\<reducer>/accessor-\<ACCESSOR_IDX>.txt*

### How to build meta-classifier/meta-regressor
1) Run [dataset.py](dataset.py) to generate pre-meta-dataframe with distances between measures and accessors orderings.
This dataframe contains references to data only, without numerical meta-features.
Note, that you can integrate arbitrary amount of internal measures here without re-estimating by accessors.
This step produces *pre-meta-dataset.json* for all accessor-estimated data. 
2) Run [features.py](features.py) to build meta-features for each 2D data representation.
Note, that you can build another meta-description here without re-estimating by accessors.
This step produces *data/\<data-name>/\<reducer>/features.txt* for all data.
3) Run [predictor.py](predictor.py) to build meta-classifier (meta-regressor) for meta-data obtained in previous steps
