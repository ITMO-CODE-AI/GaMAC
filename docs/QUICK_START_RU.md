### Подготовка табличных данных

Начнём с простейшего случая, когда набор данных состоит только из числовых признаков.
Есть некоторые предварительные условия, которые пользователь должен выполнить:

* Подключить [heaven.jar](../bin/heaven.jar) в Spark сессию;
* Выбрать директорию чекпойнтов для внутренних нужд Spark;
* Использовать функции Pyspark для чтения данных.

```python
import pyspark

DATA_PATH  =  ...
CHECKPOINT_DIR  =  ...

conf = pyspark.SparkConf()\
    .setAppName('sparkling-unimodal') \
    .setMaster('local-cluster[4, 2, 1024]') \
    .set("spark.jars", "bin/heaven.jar")

sc = pyspark.SparkContext.getOrCreate(conf)
sc.setCheckpointDir(CHECKPOINT_DIR)
sql = pyspark.SQLContext(sc)

df = sql.read.csv(f'{DATA_PATH}.csv', inferSchema=True, header=True)
```

**[SparklingBuilder](../sparkling/data/builder.py)** является входной точкой для [препроцессинга](GLOSSARY_RU.md#препроцессинг).
В нём реализованы методы для задания модальностей набора данных и их параметров (см. [документацию](../sparkling/data/builder.py)).

Чтобы «построить» табличную модальность, вам необходимо указать имена всех колонок, содержащих признаки объектов.

В примере предполагается, что данные размечены, колонка *class* обозначает внешнюю метку, а остальные колонки обозначают признаки:

```python
from sparkling import SparklingBuilder

df =  ...
name =  '...'  # имя новой колонки, которая будет содержать собранные признаки
feature_cols = set(df.columns) - {'class'}
sparkling_df = SparklingBuilder(df, class_col='class') \
    .tabular(name, feature_cols) \
    .create()
```

**[SparklingBuilder](../sparkling/data/builder.py)** - ленивый, то есть запускает вычисления только при вызове **.create()**. Он возвращает 
**[SparklingDF](../sparkling/data/dataframe.py)** - [подготовленный набор данных](GLOSSARY_RU.md#подготовленный-набор-данных),
который будет в дальнейшем использоваться.

Вы можете получить доступ к нативному датафрейму следующим образом:

```python
sparkling_df =  ...
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

### Запуск [процесса оптимизации](GLOSSARY_RU.md#процесс-оптимизации)

После получения [подготовленного набора данных](GLOSSARY_RU.md#подготовленный-набор-данных) 
(**[SparklingDF](../sparkling/data/dataframe.py)**), Вы можете настроить и 
запустить [процесс оптимизации](GLOSSARY_RU.md#процесс-оптимизации):

```python
from sparkling import Sparkling, Internal

sparkling_df =  ...
optimiser =  Sparkling(sparkling_df, measure=Internal.CALINSKI_HARABASZ)
optimal = optimiser.run(time_limit=150)
print(optimal)

# {"value": -3171.1437458030446, "algorithm": "BisectingKMeans", "params": {"k": 29, "max_iterations": 18, "min_cluster_size": 0.6529423770130484, "convergence": 1e-07, "seed": 923036730}}
```

Здесь выбрана мера *calinkski-harabasz* как [целевая](GLOSSARY_RU.md#целевая-мера), таким образом, ожидается, ч
то результат будет представлять собой алгоритм кластеризации и его гиперпараметры, максимизирующие выбранную меру.

Выделяется 150 секунд для исследования [пространства поиска](GLOSSARY_RU.md#пространство-поиска) 
по умолчанию (поскольку в данном примере оно не было явно задано, см. 
[дополнительные настройки](#дополнительная-настройка-процесса-оптимизации)).

Возвращаемый объект содержит алгоритм кластеризации с полученными гиперпараметрами, 
обученную модель кластеризации и размеченный датафрейм:

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
  

### Оценка мерами качества

Если есть размеченный датафрейм, то для него можно посчитать любую внутреннюю меру. Более того, 
если в исходном датафрейме были представлены внешние метки (и **class_col** передавался в **SparklingBuilder**), 
то можно вычислить также и внешние меры качества следующим образом:

```python
from sparkling import External, Internal

optimiser =  ...
optimal = optimiser.run(...)

External.F1.evaluate(optimal.label_sdf) # 0.5227979978880252
Internal.SF.evaluate(optimal.label_sdf) # 0.5409167557013761
```

*Полный код представлен в [examples/local/tabular.py](../examples/local/tabular.py)*

### Кластеризация изображений

**Sparkling** использует [модели глубокого обучения](GLOSSARY_RU.md#модель-глубокого-обучения), 
чтобы получить векторные представления (эмбеддинги) для изображений. Для этого нужно:

* Установить фреймворк для глубокого обучения (в примере[pytorch](https://pytorch.org/), 
см. также [requirements/pytorch.txt](../requirements/pytorch.txt));
* Включить функционал [Pyarrow](https://arrow.apache.org/docs/python/index.html) в Spark сессии;
* DataFrame должен содержать пути к изображениям относительно *root*.

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

Следующим шагом необходимо указать [модель глубокого обучения](GLOSSARY_RU.md#модель-глубокого-обучения), 
которая будет использоваться для получения эмбеддингов. Список предобученных моделей лежит 
в **[TorchImages](../sparkling/emb/torch/torch_image.py)** (другие на текущий момент не поддерживаются).

Напоминаем, что вам необходим **[SparklingBuilder](../sparkling/data/builder.py)** 
для запуска [препроцессига](GLOSSARY_RU.md#препроцессинг):

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

После этого вы можете запустить [процесс оптимизации](GLOSSARY_RU.md#процесс-оптимизации), 
как было показано в [предыдущем примере](#запуск-процесса-оптимизации).

*Полный код представлен в [examples/local/image.py](../examples/local/image.py)*

### Кластеризация текстов и мультимодальных данных

Предварительная обработка текста практически такая же, как и для изображений. Тексты должны храниться как строки 
непосредственно в колонке датафрейма. Их следует обрабатывать одной из моделей **[TorchTexts](../sparkling/emb/torch/torch_text.py)**.

Ниже представлен пример того, как, заодно, указать несколько модальностей в **SparklingBuilder**:

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

*Полный код представлен в [examples/local/text.py](../examples/local/text.py).*

### Сериализация [подготовленного набора данных](GLOSSARY_RU.md#подготовленный-набор-данных)

Получение [подготовленного набора данных](GLOSSARY_RU.md#подготовленный-набор-данных) может занять много времени, 
особенно когда присутствуют изображения/тексты, которые должны быть обработаны "тяжелыми" 
[моделями глубокого обучения](GLOSSARY_RU.md#модель-глубокого-обучения).
 
Поэтому вы можете сохранить [подготовленный набор данных](GLOSSARY_RU.md#подготовленный-набор-данных), 
а затем загрузить его, чтобы запустить новый [процесс оптимизации](GLOSSARY_RU.md#процесс-оптимизации).

**[SparklingDF](../sparkling/data/dataframe.py)** не предоставляет специального метода для этих целей, 
потому что пользователь может предпочесть разные способы хранения информации. Понадобится выполнить два шага:

* Загрузите *sparkling_df.df* (Spark датафрейм) в хранилище в любом формате, поддерживающем типы 
* **[pyspark.ml.linalg.Vector](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.linalg.Vector.html)**;
* Сохраните *sparkling_df.modals_dict* ([мета-информацию набора данных](GLOSSARY_RU.md#мета-информация-набора-данных)).

Чтобы восстановить мультимодальный датафрейм, используйте статический метод **SparklingDF.create()**. 
Вот пример сохранения метаинформации в виде JSON-файла и записи датафрейма в формате parquet:

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

*Полный код представлен в [examples/local/serialization.py](../examples/local/serialization.py).*

### Дополнительная настройка [процесса оптимизации](GLOSSARY_RU.md#процесс-оптимизации)

У [процесса оптимизации](GLOSSARY_RU.md#процесс-оптимизации) есть [собственные гиперпараметры](../sparkling/opt/main.py), 
которые можно изменить:

* *configs* - ручная настройка [пространства поиска](GLOSSARY_RU.md#пространство-поиска);
* *hyper_opt* - выбор [реализации HPO](GLOSSARY_RU.md#реализация-hpo) ([OPTUNA или SMAC](../sparkling/opt/hyperopts.py));
* *mab_solver* - выбор стратегии решения задачи о многоруком бандите ([UCB или SOFTMAX](../sparkling/opt/mabs.py));
* *measure* - явное указание [целевой меры качества](GLOSSARY_RU.md#целевая-мера) или автоматический выбор 
с помощью [алгоритма рекомендации](GLOSSARY_RU.md#алгоритм-рекомендации).
  
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

*Дополнительные спецификации системы можно найти в [APPLICATION.docx](APPLICATION.docx)*