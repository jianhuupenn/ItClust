# ItClust
ransfer learning significantly improved clustering and cell type classification in single-cell RNA-seq analysis

## ItClust: Transfer learning improves clustering and cell type classification in single-cell RNA-seq analysis

ItClust is an Iterative Transfer learning algorithm for scRNA-seq Clustering. It starts from building a training neural network to extract gene-expression signatures from a well-labeled source dataset. This step enables initializing the target network with parameters estimated from the training network. The target network then leverages information in the target dataset to iteratively fine-tune parameters in an unsupervised manner, so that the target-data-specific gene-expression signatures are captured. Once fine-tuning is finished, the target network then returns clustered cells in the target data.
ItClust has shown to be a powerful tool for scRNA-seq clustering and cell type classification analysis. It can accurately extract information from source data and apply it to help cluster cells in target data. It is robust to strong batch effect between source and target data, and is able to separate unseen cell types in the target. Furthermore, it provides confidence scores that facilitates cell type assignment. With the increasing popularity of scRNA-seq in biomedical research, we expect ItClust will make better utilization of the vast amount of existing well annotated scRNA-seq datasets, and enable researchers to accurately cluster and annotate cells in scRNA-seq.


![ItClust workflow](./docs/assets/images/desc_workflow.png)

For thorough details, see the preprint: [Bioxiv](https://www.biorxiv.org/content/10.1101/530378v1?rss=1)
<br>


## Usage

The [**desc**](https://github.com/eleozzr/desc) package is an implementation of deep embedding for single-cell clustering. With desc, you can:

- Preprocess single cell gene expression data from various formats.
- Build a low-dimensional representation of the single-cell gene expression data.
- Obtain soft-clustering assignments of cells.
- Visualize the cell clustering results and  the  gene expression patterns.

<br>

# Installation

To install  `desc` package you must make sure that your python version is either  `3.5.x` or `3.6.x`. If you don’t know the version of python you can check it by:
```python
>>>import platform
>>>platform.python_version()
#3.5.3
>>>import tensorflow as tf
>>> tf.__version__
#1.7.0
```
**Note:** Because desc depend on `tensorflow`, you should make sure the version of `tensorflow` is lower than `2.0` if you want to get the same results as the results in our paper.
Now you can install the current release of `desc` by the following three ways.

* PyPI  
Directly install the package from PyPI.

```bash
$ pip install desc
```
**Note**: you need to make sure that the `pip` is for python3，or we should install desc by
```bash 
python3 -m pip install desc 
#or
pip3 install desc
```

If you do not have permission (when you get a permission denied error), you should install desc by 

```bash
$ pip install --user desc
```

* Github  
Download the package from [Github](https://github.com/eleozzr/desc) and install it locally:

```bash
git clone https://github.com/eleozzr/desc
cd desc
pip install .
```

* Anaconda

If you do not have  Python3.5 or Python3.6 installed, consider installing Anaconda  (see [Installing Anaconda](https://docs.anaconda.com/anaconda/install/)). After installing Anaconda, you can create a new environment, for example, `DESC` (*you can change to any name you like*):

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

Please check desc [Tutorial](https://eleozzr.github.io/desc/tutorial.html) for more details. And we also provide a simple example [paul\_desc.md](./desc_paul.md) or [desc\_paul.ipynb](./desc_paul.ipynb) for reproducing the results of Paul's data in our paper.


<br>

## Contributing

Souce code: [Github](https://github.com/eleozzr/desc)  

We are continuing adding new features. Bug reports or feature requests are welcome.

<br>


## References

Please consider citing the following reference:

- Xiangjie Li, Yafei Lyu, Jihwan Park, Jingxiao Zhang, Dwight Stambolian, Katalin Susztak, Gang Hu, Mingyao Li. Deep learning enables accurate clustering and batch effect removal in single-cell RNA-seq analysis. 2019. bioRxiv 530378; doi: [https://doi.org/10.1101/530378](https://www.biorxiv.org/content/10.1101/530378v1?rss=1)
<br>
