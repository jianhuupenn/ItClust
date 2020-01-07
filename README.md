# ItClust

## ItClust: Transfer learning improves clustering and cell type classification in single-cell RNA-seq analysis

ItClust is an Iterative Transfer learning algorithm for scRNA-seq Clustering. It starts from building a training neural network to extract gene-expression signatures from a well-labeled source dataset. This step enables initializing the target network with parameters estimated from the training network. The target network then leverages information in the target dataset to iteratively fine-tune parameters in an unsupervised manner, so that the target-data-specific gene-expression signatures are captured. Once fine-tuning is finished, the target network then returns clustered cells in the target data.
ItClust has shown to be a powerful tool for scRNA-seq clustering and cell type classification analysis. It can accurately extract information from source data and apply it to help cluster cells in target data. It is robust to strong batch effect between source and target data, and is able to separate unseen cell types in the target. Furthermore, it provides confidence scores that facilitates cell type assignment. With the increasing popularity of scRNA-seq in biomedical research, we expect ItClust will make better utilization of the vast amount of existing well annotated scRNA-seq datasets, and enable researchers to accurately cluster and annotate cells in scRNA-seq.

![ItClust workflow](docs/asserts/images/workflow.jpg)

For thorough details, see the preprint: [Bioxiv]()
<br>


## Usage

The [**ItClust**](https://github.com/jianhuupenn/ItClust) package is an implementation of Iterative Transfer learning algorithm for scRNA-seq Clustering. With ItClust, you can:

- Preprocess single cell gene expression data from various formats.
- Build a network for target data clustering with prioe knowledge learnt from the source data
- Obtain soft-clustering assignments of cells.
- Obtain celltype confidence score for each clsuter to assist celltype assignment.
- Visualize the cell clustering/classification results and the gene expression patterns.

<br>

# Installation

To install  `ItClust` package you must make sure that your python version is either  `3.5.x` or `3.6.x`. If you don’t know the version of python you can check it by:
```python
import platform
platform.python_version()
#3.5.3
```
**Note:** Because desc depend on `tensorflow`, you should make sure the version of `tensorflow` is lower than `2.0` if you want to get the same results as the results in our paper.
```
import tensorflow as tf
tf.__version__
#1.7.0
```
Now you can install the current release of `ItClust` by the following three ways.

* PyPI  
Directly install the package from PyPI.

```bash
pip install ItClust
```
**Note**: you need to make sure that the `pip` is for python3，or we should install desc by
```bash 
python3 -m pip install ItClust
#or
pip3 install ItClust
```

If you do not have permission (when you get a permission denied error), you should install desc by 

```bash
pip install --user  ItClust
```

* Github  
Download the package from [Github](https://github.com/eleozzr/desc) and install it locally:

```bash
git clone https://github.com/eleozzr/desc
cd desc
pip install .
```

* Anaconda

If you do not have  Python3.5 or Python3.6 installed, consider installing Anaconda  (see [Installing Anaconda](https://docs.anaconda.com/anaconda/install/)). After installing Anaconda, you can create a new environment, for example, `ItClust` (*you can change to any name you like*):

```bash
conda create -n DESC python=3.5.3
# activate your environment 
source activate DESC 
git clone https://github.com/eleozzr/desc
cd desc
python setup.py build
python setup.py install
# now you can check whether `desc` installed successfully!
```

Please check desc [Tutorial]() for more details. And we also provide a simple example [paul\_desc.md](./desc_paul.md) or [desc\_paul.ipynb](./desc_paul.ipynb) for reproducing the results of Paul's data in our paper.


<br>


# Read in data
The current version of ItClust works with an AnnData object. AnnData stores a data matrix .X together with annotations of observations .obs, variables .var and unstructured annotations .uns. The ItClust package provides 3 ways to prepare an AnnData object for the following analysis.
<br>
1.1 Start from a 10X dataset
Here we use the pbmc data as an example: Download the data and unzip it. Then move everything in filtered_gene_bc_matrices/hg19/ to data/pbmc/.
```python
adata = read_10X(data_path='./data/pbmc')
#var_names are not unique, "make_index_unique" has applied
```
1.2 Start from .mtx and .tsv files
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
<br>

1.3 Start from a *.h5ad file
We will use human pancreas data as our example for transfer learning. The Baron et al. data is used as source data and Segerstolpe et al. is treated as traget data. We can use following code to read data in from *.h5ad files:
```python
import scanpy.api as sc
adata_train=sc.read("./data/pancreas/Bh.h5ad")
adata_test=sc.read("./data/pancreas/smartseq2.h5ad")
```

## Clustering
```python
import ItClust
clf=ItClust.transfer_learning_clf()
clf.fit(source_data=adata_train, target_data=adata_test)
clf.predict(save_dir="./pancreas_results")

```
## Contributing

Souce code: [Github](https://github.com/jianhuupenn/ItClust)  

We are continuing adding new features. Bug reports or feature requests are welcome.

<br>


## References

Please consider citing the following reference:

- 
<br>
