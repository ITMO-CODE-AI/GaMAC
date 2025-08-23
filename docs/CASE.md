[![README](https://img.shields.io/badge/README-md-blue.svg)](../README.md)
[![ru](https://img.shields.io/badge/lang-ru-red.svg)](CASE_RU.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](CASE.md)

Applications are provided in [notebooks](https://github.com/ITMO-CODE-AI/GaMAC/blob/develop/notebooks/examples/example_on_realdata.ipynb)

### Clustering for Industrial Log Analysis  

Log clustering aims to automatically group log entries associated with common event sources, such as application processes, system events, or runtime environments. The primary goal is to uncover hidden patterns in the data that indirectly reveal the nature of log generation, even if the log format does not explicitly indicate the source. For example, logs from three different applications running in two environments (development and production) should be separated into clusters corresponding to these applications and environments. A key challenge is assessing algorithm relevance: how accurately they identify real existing groups rather than creating artificial divisions.  

The data for analysis typically consists of unstructured or semi-structured text entries. Their synthetic nature (in test datasets) simplifies clustering quality verification since the true log sources are known in advance. However, in real-world scenarios, metadata such as timestamps, event types (errors, warnings), keywords (e.g., "timeout," "connection failed"), or execution context (process IDs, software versions) become the basis for feature extraction. These features can be either explicit or latent, requiring transformation into numerical vectors for machine learning algorithms.  

Example application: When analyzing logs from three applications (A, B, C) in two environments (Dev, Prod), a combination of methods is used. First, HDBSCAN is applied to automatically determine the number of clusters, avoiding assumptions about the number of sources. Then, logs are filtered by keywords related to environments (e.g., "Dev," "Prod") to split clusters into subgroups. Visualization using dimensionality reduction techniques such as t-SNE or UMAP helps analyze data structure and verify whether clusters match expected sources.  

### Clustering for RGB Image Color Analysis  

RGB image color clustering aims to automatically group pixels with similar color values into separate clusters, where each cluster represents an averaged color or dominant shade from the corresponding group. The primary goal is to simplify image representation by extracting key color patterns, which can be useful for tasks such as data compression, object segmentation, noise reduction, or color palette analysis. For example, an image with a red gradient containing hundreds of shades can be reduced to a few clusters representing the primary tones. A key challenge is determining how well algorithms adapt to color variations: from precisely distinguishing closely related shades (e.g., scarlet and burgundy) to merging them into a single cluster when necessary.
