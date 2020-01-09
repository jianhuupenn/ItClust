<h1><center>ItClust Tutorial</center></h1>


<center>Author: Jian Hu, Xiangjie Li, Gang Hu, Yafei Lyu, Katalin Susztak, Mingyao Li*</center>

### 0. Import python modules


```python
import ItClust as ic
import scanpy.api as sc
import os
from numpy.random import seed
from tensorflow import set_random_seed
import pandas as pd
import numpy as np
import warnings
os.environ["CUDA_VISIBLE_DEVICES"]="1"
warnings.filterwarnings("ignore")
#import sys
#!{sys.executable} -m pip install 'scanpy==1.4.4.post1'
#Set seeds
seed(20180806)
np.random.seed(10)
set_random_seed(20180806) # on GPU may be some other default

```

    Using TensorFlow backend.
    /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/scanpy/api/__init__.py:6: FutureWarning: 
    
    In a future version of Scanpy, `scanpy.api` will be removed.
    Simply use `import scanpy as sc` and `import scanpy.external as sce` instead.
    
      FutureWarning


### 1. Read in data
The current version of ItClust works with an AnnData object. AnnData stores a data matrix .X together with annotations of observations .obs, variables .var and unstructured annotations .uns. The ItClust package provides 3 ways to prepare an AnnData object for the following analysis.
<br>
ItClust support most forms of the scRNAseq data, including UMI, TPM, FPKM.
<br>
<br>
Important Note: For the source data, please store the true celltype label information in one column named "celltype".

#### 1.1 Start from a 10X dataset
Here we use the pbmc data as an example:
Download the data and unzip it. Then move everything in filtered_gene_bc_matrices/hg19/ to data/pbmc/.


```python
adata = read_10X(data_path='./data/pbmc')
```

    var_names are not unique, "make_index_unique" has applied


#### 1.2  Start from *.mtx and *.tsv files
When the expression data do not follow the standard 10X dataset format, we can manually import the data as follows.


```python
#1 Read the expression matrix from *.mtx file.
# The row of this matrix correspond to cells, columns corresond to genes. 
adata = read_mtx('./data/pbmc/matrix.mtx').T 

#2 Read the *.tsv file for gene annotations. Make sure the gene names are unique.
genes = pd.read_csv('./data/pbmc/genes.tsv', header=None, sep='\t')
adata.var['gene_ids'] = genes[0].values
adata.var['gene_symbols'] = genes[1].values
adata.var_names = adata.var['gene_symbols']
# Make sure the gene names are unique
adata.var_names_make_unique(join="-")

#3 Read the *.tsv file for cell annotations. Make sure the cell names are unique.
cells = pd.read_csv('./data/pbmc/barcodes.tsv', header=None, sep='\t')
adata.obs['barcode'] = cells[0].values
adata.obs_names = cells[0]
# Make sure the cell names are unique
adata.obs_names_make_unique(join="-")
```

    Variable names are not unique. To make them unique, call `.var_names_make_unique`.


#### 1.3 Start from a *.h5ad file
We will use human pancreas data as our example for transfer learning.
The Baron et al. data is used as source data and Segerstolpe et al. is treated as traget data. We can use following code to read data in from *.h5ad files:


```python
adata_train=sc.read("./data/pancreas/Bh.h5ad")
adata_test=sc.read("./data/pancreas/smartseq2.h5ad")
```

### 2. Fit ItClust model
ItClust includes preprocessing steps, that is, filtering of cells/genes, normalization, scaling and selection of highly variables genes.


```python
clf=ic.transfer_learning_clf()
clf.fit(adata_train, adata_test)
```

    /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/scanpy/preprocessing/_simple.py:284: DeprecationWarning: Use is_view instead of isview, isview will be removed in the future.
      if isinstance(data, AnnData) and data.isview:


    the var_names of adata.raw: adata.raw.var_names.is_unique=: True


    /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/scanpy/preprocessing/_simple.py:284: DeprecationWarning: Use is_view instead of isview, isview will be removed in the future.
      if isinstance(data, AnnData) and data.isview:
    /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/scanpy/utils.py:618: DeprecationWarning: Use is_view instead of isview, isview will be removed in the future.
      if adata.isview:
    /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/scanpy/preprocessing/_simple.py:284: DeprecationWarning: Use is_view instead of isview, isview will be removed in the future.
      if isinstance(data, AnnData) and data.isview:


    the var_names of adata.raw: adata.raw.var_names.is_unique=: True


    /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/scanpy/preprocessing/_simple.py:284: DeprecationWarning: Use is_view instead of isview, isview will be removed in the future.
      if isinstance(data, AnnData) and data.isview:
    /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/scanpy/utils.py:618: DeprecationWarning: Use is_view instead of isview, isview will be removed in the future.
      if adata.isview:


    The number of training celltypes is:  14
    Training the source network
    The layer numbers are[32, 16]
    The shape of xtrain is:8569:867
    The shape of xtest is:2394:867
    Doing DEC: pretrain
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:133: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    ...Pretraining...
    Doing SAE: pretrain_stacks
    Pretraining the 1th layer...
    learning rate = 0.1
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:973: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:2741: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:174: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:181: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:190: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:199: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.
    


    /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/tensorflow_core/python/framework/indexed_slices.py:339: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3,and in 3.9 it will stop working
      if not isinstance(values, collections.Sequence):


    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:206: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.
    
    learning rate = 0.01
    learning rate = 0.001
    The 1th layer has been pretrained.
    Pretraining the 2th layer...
    learning rate = 0.1
    learning rate = 0.01
    learning rate = 0.001
    The 2th layer has been pretrained.
    Doing SAE: pretrain_autoencoders
    Copying layer-wise pretrained weights to deep autoencoders
    Fine-tuning autoencoder end-to-end
    learning rate = 0.1
    learning rate = 0.010000000000000002
    learning rate = 0.001
    learning rate = 0.0001
    learning rate = 1e-05
    learning rate = 1.0000000000000002e-06
    Pretraining time:  158.4946711063385
    y known, initilize Cluster centroid using y
    The shape of cluster_center is (14, 16)
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:1521: The name tf.log is deprecated. Please use tf.math.log instead.
    
    Doing DEC: fit_supervised
    WARNING:tensorflow:From /Users/hujian1/anaconda3/envs/keras/lib/python3.7/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    Training model finished! Start to fit target network!
    Doing DEC: pretrain_transfer
    The shape of features is (2394, 16)
    The shape of y_trans is (2394,)
    ...predicted y_test known, use it to get n_cliusters and init_centroid
    The length layers  of self.model 4
    Doing DEC: fit_trajectory
    The value of delta_label of current 1 th iteration is 0.002506265664160401 >= tol [0.001]
    This is the iteration of  0
    The value of delta_label of current 2 th iteration is 0.004177109440267335 >= tol [0.001]
    The value of delta_label of current 3 th iteration is 0.001670843776106934 >= tol [0.001]
    delta_label  0.000835421888053467 < tol  [0.001]
    Reached tolerance threshold. Stopped training.
    The final prediction cluster is:
    2     988
    5     457
    3     323
    8     210
    0     182
    4     126
    1      56
    6      22
    10      8
    9       7
    7       7
    11      6
    12      2
    dtype: int64
    How many trajectories  1


### 3. Prediction
predict() function will return the cluster prediction, clustering probability matrix and cell type confidence score. 

If the parameter write==True(default), it will also write the results to save_dir

The cluster prediction is written to save_dir+"/clustering_results.csv".

The cell type confidence score is written to save_dir+"/celltype_assignment.txt"

The clustering probability matrix is written to save_dir+"/clustering_prob.csv"



```python
pred, prob, celltype_pred=clf.predict()
pred.head()
```

    Results are written to  ./results





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cell_id</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AZ_A2-target</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AZ_H5-target</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ_G5-target</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AZ_D8-target</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AZ_D12-target</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



### 4. Visualization
#### 4.1 t-SNE


```python
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['figure.dpi']= 300
colors_use=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896','#bec1d4','#bb7784','#4a6fe3','#FFFF00''#111010']
# Run t-SNE
clf.adata_test.obsm['X_tsne']=clf.tSNE()
num_celltype=len(clf.adata_test.obs["celltype"].unique())
clf.adata_test.uns["celltype_colors"]=list(colors_use[:num_celltype])
clf.adata_test.uns["decisy_trans_True_colors"]=list(colors_use[:num_celltype])
sc.pl.tsne(clf.adata_test,color=["decisy_trans_True","celltype"],title=["ItClust predition","True cell type"],show=True,size=50000/clf.adata_test.shape[0])
```

    Doing t-SNE!
    WARNING: Consider installing the package MulticoreTSNE (https://github.com/DmitryUlyanov/Multicore-TSNE). Even for n_jobs=1 this speeds up the computation considerably and might yield better converged results.



![png](output_15_1.png)


#### 5.2 U-map


```python
clf.adata_test.obsm['X_umap']
sc.pl.umap(clf.adata_test,color=["decisy_trans_True","celltype"],show=True,save=None,title=["ItClust predition","True cell type"],size=50000/adata_test.shape[0])
```


![png](output_17_0.png)

