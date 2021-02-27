# ItClust

## ItClust: Iterative transfer learning with neural network improves clustering and cell type classification in single-cell RNA-seq analysis
ItClust is an Iterative Transfer learning algorithm for scRNA-seq Clustering. It starts from building a training neural network to extract gene-expression signatures from a well-labeled source dataset. This step enables initializing the target network with parameters estimated from the training network. The target network then leverages information in the target dataset to iteratively fine-tune parameters in an unsupervised manner, so that the target-data-specific gene-expression signatures are captured. Once fine-tuning is finished, the target network then returns clustered cells in the target data.
ItClust has shown to be a powerful tool for scRNA-seq clustering and cell type classification analysis. It can accurately extract information from source data and apply it to help cluster cells in target data. It is robust to strong batch effect between source and target data, and is able to separate unseen cell types in the target. Furthermore, it provides confidence scores that facilitates cell type assignment. With the increasing popularity of scRNA-seq in biomedical research, we expect ItClust will make better utilization of the vast amount of existing well annotated scRNA-seq datasets, and enable researchers to accurately cluster and annotate cells in scRNA-seq.

![ItClust workflow](docs/asserts/images/workflow.jpg)

For thorough details, see the preprint: [Bioxiv]()
<br>

## Usage

The [**ItClust**](https://github.com/jianhuupenn/ItClust) package is an implementation of Iterative Transfer learning algorithm for scRNA-seq Clustering. With ItClust, you can:

- Preprocess single cell gene expression data from various formats.
- Build a network for target data clustering with prioe knowledge learnt from the source data.
- Obtain soft-clustering assignments of cells.
- Obtain cell type confidence score for each clsuter to assist cell type assignment.
- Visualize cell clustering/classification results and gene expression patterns.

## System Requirements
Python support packages: pandas, numpy, keras, scipy, scanpy, anndata, natsort, sklearn.

## Versions the software has been tested on
Environment 1:
- System: Mac OS 10.13.6
- Python: 3.7.0
- Python packages: pandas = 0.25.3, numpy = 1.18.1, keras = 2.2.4, tensorflow = 1.14.0, scipy = 1.4.1, scanpy = 1.4.4.post1, anndata = 0.6.22.post1, natsort = 7.0.1, sklearn = 0.22.1

Environment 2:
- System: Linux 3.10.0
- Python: 3.7.5
- Python packages: pandas = 0.25.3, numpy = 1.17.3, keras = 2.3.1, scipy = 1.4.1, scanpy = 1.4.4.post1, anndata = 0.6.22.post1, natsort = 6.0.0, sklearn = 0.21.3

Environment 3:
- System: Ubuntu 16.04.6 LTS
- Python: 3.5.2
- Python packages: pandas = 0.22.0, numpy = 1.16.4, keras = 2.2.4, scipy = 1.0.1, scanpy = 1.3.1+21.g1df151f, anndata = 0.6.20, natsort = 5.2.0, sklearn = 0.19.1

## Installation guide, Demo and Instructions
Please please refer to: https://github.com/jianhuupenn/ItClust/blob/master/tutorial/tutorial.md


## Contributing

Souce code: [Github](https://github.com/jianhuupenn/ItClust)  
Author email: jianhu@pennmedicine.upenn.edu
<br>
We are continuing adding new features. Bug reports or feature requests are welcome.
<br>

## Debugging

If you see this error information: 
#### TypeError: add_weight() got multiple values for argument 'name'
Please update your Keras and TensorFlow to the same version as one of the tested environments.

## Reference

Please consider citing the following reference:
- Hu, J., Li, X., Hu, G. et al. Iterative transfer learning with neural network for clustering and cell type classification in single-cell RNA-seq analysis. Nat Mach Intell (2020). https://doi.org/10.1038/s42256-020-00233-7
- https://www.nature.com/articles/s42256-020-00233-7
- If you do not have access to the link above, you can access the paper by https://rdcu.be/b78Ya
- https://www.biorxiv.org/content/10.1101/2020.02.02.931139v1.full (preprint)
