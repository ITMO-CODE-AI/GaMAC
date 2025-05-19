### Preparing tabular data

Let's start with the simplest case, when you have fully numeric dataset. 
There are some prerequisites you need to satisfy:

* Include [heaven.jar](../bin/heaven.jar) into Spark session;
* Set checkpoint dir for Spark internal needs;
* Use native Pyspark functions to read dataframe.

```python
import pyspark

DATA_PATH = ...
CHECKPOINT_DIR = ...

conf = pyspark.SparkConf()\
    .setAppName('sparkling-unimodal') \
    .setMaster('local-cluster[4, 2, 1024]') \
    .set("spark.jars", "bin/heaven.jar")

sc = pyspark.SparkContext.getOrCreate(conf)
sc.setCheckpointDir(CHECKPOINT_DIR)
sql = pyspark.SQLContext(sc)

df = sql.read.csv(f'{DATA_PATH}.csv', inferSchema=True, header=True)
```

**[SparklingBuilder](../sparkling/data/builder.py)** is the entry point for 
[preprocessing pipeline](GLOSSARY.md#preprocessing-pipeline). It implements methods to 
define dataframe modalities and their parameters (see [docs](../sparkling/data/builder.py)).

To "build" tabular modality, you need to specify all feature columns of dataframe.

Here we assume that dataframe has external label (column *class*) and all other columns are features:

```python
from sparkling import SparklingBuilder

df = ...
name = '...'  # new column name, that will contain assembled values
feature_cols = set(df.columns) - {'class'}
sparkling_df = SparklingBuilder(df, class_col='class') \
    .tabular(name, feature_cols) \
    .create()
```

**[SparklingBuilder](../sparkling/data/builder.py)** is lazy evaluator and starts execution only on **.create()** invocation. It returns 
**[SparklingDF](../sparkling/data/dataframe.py)** [preprocessed dataframe](GLOSSARY.md#preprocessed-dataframe), which will be used further. 

You can access underlying Spark dataframe:

```python
sparkling_df = ...
sparkling_df.df.show()

# +------+----+--------------------+
# |_class| _id|            features|
# +------+----+--------------------+
# |     3|3883|[-0.5098039215686...|
# |     3| 881|[-0.4509803921568...|
# |     2|2709|[-0.6666666666666...|
# |     3|4461|[-0.6078431372549...|
# |     3|1368|[-0.7254901960784...|
# |     3|2599|[-0.5294117647058...|
# |     3|4582|[-0.4313725490196...|
# |     3|2277|[-0.8823529411764...|
# |     2|1418|[-0.4705882352941...|
# |     3|4715|[-0.8627450980392...|
# |     3|1600|[-0.5882352941176...|
# |     3|1548|[-0.4509803921568...|
# |     3|3122|[-0.8039215686274...|
# |     4| 968|[-1.0,-0.48076923...|
# |     3|2003|[-0.5686274509803...|
# |     3|2949|[-0.6078431372549...|
# |     3|3146|[-0.7058823529411...|
# |     3|1663|[-0.5098039215686...|
# |     3|4143|[-0.6666666666666...|
# |     2|3296|[-0.6078431372549...|
# +------+----+--------------------+
# only showing top 20 rows
```

### Run [optimisation pipeline](GLOSSARY.md#optimisation-pipeline)

After you obtain [preprocessed dataframe](GLOSSARY.md#preprocessed-dataframe) 
(**[SparklingDF](../sparkling/data/dataframe.py)**), you can configure and 
start [optimisation pipeline](GLOSSARY.md#optimisation-pipeline):

```python
from sparkling import Sparkling, Internal

sparkling_df = ...
optimiser = Sparkling(sparkling_df, measure=Internal.CALINSKI_HARABASZ)
optimal = optimiser.run(time_limit=150)
print(optimal)

# {"value": -3171.1437458030446, "algorithm": "BisectingKMeans", "params": {"k": 29, "max_iterations": 18, "min_cluster_size": 0.6529423770130484, "convergence": 1e-07, "seed": 923036730}}
```

Here we choose *calinkski-harabasz* as [target measure](GLOSSARY.md#target-measure), so the result is
expected to represent clustering algorithm and its hyperparameters that maximises the measure.

We give 150 seconds to explore default [search space](GLOSSARY.md#search-space) (as we 
do not configure it explicitly, see [customisation](#customising-optimisation-pipeline)).

The result object contains clustering algorithm with defined hyperparameters,
fitted clustering model and dataframe with labels:

```python
optimal = ...

print(optimal.algo)  # BisectingKMeans
print(optimal.model)  # BisectingKMeansModel
print(optimal.value)  # -3171.1437458030446, target measure value (minimised)

optimal.label_sdf.df.show()

# +------+----+--------------------+------+
# |_class| _id|            features|_label|
# +------+----+--------------------+------+
# |     3|3883|[-0.5098039215686...|     1|
# |     3| 881|[-0.4509803921568...|     0|
# |     2|2709|[-0.6666666666666...|     0|
# |     3|4461|[-0.6078431372549...|     1|
# |     3|1368|[-0.7254901960784...|     0|
# |     3|2599|[-0.5294117647058...|     0|
# |     3|4582|[-0.4313725490196...|     0|
# |     3|2277|[-0.8823529411764...|     1|
# |     2|1418|[-0.4705882352941...|     0|
# |     3|4715|[-0.8627450980392...|     0|
# |     3|1600|[-0.5882352941176...|     0|
# |     3|1548|[-0.4509803921568...|     1|
# |     3|3122|[-0.8039215686274...|     1|
# |     4| 968|[-1.0,-0.48076923...|     1|
# |     3|2003|[-0.5686274509803...|     1|
# |     3|2949|[-0.6078431372549...|     0|
# |     3|3146|[-0.7058823529411...|     0|
# |     3|1663|[-0.5098039215686...|     0|
# |     3|4143|[-0.6666666666666...|     1|
# |     2|3296|[-0.6078431372549...|     0|
# +------+----+--------------------+------+
# only showing top 20 rows
```

### Evaluating measures

Provided you obtain labeled dataframe, you can evaluate any internal measure on it. Moreover, if external
labels were present in original dataframe (and you pass**class_col** into **SparklingBuilder**), 
you can evaluate any external measure this way:

```python
from sparkling import External, Internal

optimiser = ...
optimal = optimiser.run(...)

External.F1.evaluate(optimal.label_sdf)  # 0.5227979978880252
Internal.SF.evaluate(optimal.label_sdf)  # 0.5409167557013761
```

*You can find full listing in [examples/local/tabular.py](../examples/local/tabular.py)*
 
### Clustering images

**Sparkling** utilises [deep learning models](GLOSSARY.md#deep-learning-model) to obtain 
vector representations (embeddings) for images, so the prerequisites are:

* Installed deep learning framework ([pytorch](https://pytorch.org/) here, 
see also [requirements/pytorch.txt](../requirements/pytorch.txt));
* Enable [Pyarrow](https://arrow.apache.org/docs/python/index.html) in Spark session;
* Dataframe should contain relative to *root* (see after-next snippet) paths to images.

```python
import pyspark

CHECKPOINT_DIR = ...

conf = pyspark.SparkConf()\
        .setAppName(f'sparkling-pytorch-image')\
        .setMaster('local-cluster[2, 2, 2048]')\
        .set("spark.sql.execution.arrow.enabled", "true")\
        .set("spark.jars", "bin/heaven.jar")\
        .setExecutorEnv('ARROW_PRE_0_15_IPC_FORMAT', '1')

ss = pyspark.sql.SparkSession.Builder().config(conf=conf).getOrCreate()
ss.sparkContext.setCheckpointDir(CHECKPOINT_DIR)
```

The next step is to specify [deep learning model](GLOSSARY.md#deep-learning-model), 
that will be used to obtain embeddings. There are a list of predefined models (others are not yet supported) 
in **[TorchImages](../sparkling/emb/torch/torch_image.py)**.

Again, you need **[SparklingBuilder](../sparkling/data/builder.py)** to run [preprocessing pipeline](GLOSSARY.md#preprocessing-pipeline):

```python
from sparkling import SparklingBuilder
from sparkling.emb.torch import TorchImages

df = ...
root = ...  # absolute path of a root directory for all images
name = '...'  # dataframe column name with images paths

sparkling_df = SparklingBuilder(df, class_col='class')\
    .image(name, TorchImages.CONVNEXT, root)\
    .create()
```

After that, you can run [optimisation pipeline](GLOSSARY.md#optimisation-pipeline) 
as from [previous example](#run-optimisation-pipeline).

*See [examples/local/image.py](../examples/local/image.py) for a full code sample*

### Clustering texts and multimodal dataframes

Preprocessing texts is nearly the same, as images. Texts should be stored as strings directly in dataframe column 
and should be processed by one of the **[TorchTexts](../sparkling/emb/torch/torch_text.py)** models.

Also, here is an example, how to specify multiple modalities in **SparklingBuilder**:

```python
from sparkling import SparklingBuilder
from sparkling.emb.torch import TorchTexts

df = ...
tabular_in_cols = {...}
tabular_out_col = '...'
text_col = '...'

sparkling_df = SparklingBuilder(df)\
        .tabular(tabular_out_col, tabular_in_cols)\
        .text(text_col, TorchTexts.BERT)\
        .create()
```

*See [examples/local/text.py](../examples/local/text.py) for a full code sample.*

### Serialize [preprocessed dataframe](GLOSSARY.md#preprocessed-dataframe)

It can take a while to obtain [preprocessed dataframe](GLOSSARY.md#preprocessed-dataframe), especially when image/text 
modalities are present and needed to be processed by heavy [deep learning models](GLOSSARY.md#deep-learning-model).

That is why you can save [preprocessed dataframe](GLOSSARY.md#preprocessed-dataframe) and then restore it to 
run another [optimisation pipeline](GLOSSARY.md#optimisation-pipeline).

**[SparklingDF](../sparkling/data/dataframe.py)** does not provide special method for this on purpose, 
because you may prefer different strategies to store information. You need two steps:

* Upload *sparkling_df.df* (Spark dataframe) to a storage in any format, that supports 
**[pyspark.ml.linalg.Vector](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.linalg.Vector.html)** types;
* Save *sparkling_df.modals_dict* ([dataframe meta](GLOSSARY.md#dataframe-meta)) into the driver.

To restore multimodal dataframe, use static method **SparklingDF.create()**. Here is an example of 
storing meta information into driver as json file and writing dataframe into parquet format

```python
import json
import pyspark

from sparkling import SparklingDF

ss = pyspark.sql.SparkSession(...)
original_sparkling_df = ...
driver_meta_path = ...
parquet_path = ...

# save modalities info on driver and dataframe on cluster
with open(driver_meta_path, 'w') as modals_fp:
    json.dump(original_sparkling_df.modal_dict, modals_fp)
original_sparkling_df.df.write.parquet(parquet_path)

# restore dataframe from distributed storage and modals from driver
serialized_df = ss.read.parquet(parquet_path)
with open(driver_meta_path) as modals_fp:
    modals = json.load(modals_fp)
sparkling_df_restored = SparklingDF.create(serialized_df, modals)
```

*See [examples/local/serialization.py](../examples/local/serialization.py) for a full code sample.*

### Customising [optimisation pipeline](GLOSSARY.md#optimisation-pipeline)

There are some [extra options](../sparkling/opt/main.py), which you can manipulate 
to customise [optimisation pipeline](GLOSSARY.md#optimisation-pipeline):

* *configs* - manually configure [search space](GLOSSARY.md#search-space);
* *hyper_opt* - switch [HPO backend](GLOSSARY.md#hpo-backend) ([OPTUNA or SMAC](../sparkling/opt/hyperopts.py));
* *mab_solver* - choose multi-armed bandit solver strategy ([UCB or SOFTMAX](../sparkling/opt/mabs.py));
* *measure* - specify [target measure](GLOSSARY.md#target-measure) or let the framework pick 
[target measure](GLOSSARY.md#target-measure) by using [CVIPredictor](GLOSSARY.md#cvi-predictor).

```python
from sparkling import Sparkling, Internal
from sparkling.opt import MabSolver, HyperOpt
from sparkling.algorithms import KMeansConf, MeanShiftConf

sparkling_df = ...
kmeans_conf = KMeansConf(
    k=(3, 8),  # Search from 3 to 8 clusters inclusive
    max_iterations=20  # Constantly use 20 as upper bound for algorithm iterations
)
mean_shift_conf = MeanShiftConf(
    radius=(0.1, 0.4)  # Search neighbourhood radius in that continuous range
)

optimiser = Sparkling(
    sparkling_df, 
    configs=[kmeans_conf, mean_shift_conf],
    mab_solver=MabSolver.UCB,
    hyper_opt=HyperOpt.OPTUNA,
    measure=Internal.GD41_APPROX  # Or None, to invoke CVIPredictor
)
```

*You can find extra API information in [APPLICATION.docx](APPLICATION.docx)*