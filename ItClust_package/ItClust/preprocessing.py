import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import os
from anndata import AnnData,read_csv,read_text,read_mtx
from scipy.sparse import issparse
from natsort import natsorted
from anndata import read_mtx
from anndata.utils import make_index_unique


def read_10X(data_path, var_names='gene_symbols'):
    adata = read_mtx(data_path + '/matrix.mtx').T
    genes = pd.read_csv(data_path + '/genes.tsv', header=None, sep='\t')
    adata.var['gene_ids'] = genes[0].values
    adata.var['gene_symbols'] = genes[1].values
    assert var_names == 'gene_symbols' or var_names == 'gene_ids', \
        'var_names must be "gene_symbols" or "gene_ids"'
    if var_names == 'gene_symbols':
        var_names = genes[1]
    else:
        var_names = genes[0]
    if not var_names.is_unique:
        var_names = make_index_unique(pd.Index(var_names))
        print('var_names are not unique, "make_index_unique" has applied')
    adata.var_names = var_names
    cells = pd.read_csv(data_path + '/barcodes.tsv', header=None, sep='\t')
    adata.obs['barcode'] = cells[0].values
    adata.obs_names = cells[0]
    return adata



def change_to_continuous(q):
    #y_trans=q.argmax(axis=1)
    y_pred=np.asarray(np.argmax(q,axis=1),dtype=int)
    unique_labels=np.unique(q.argmax(axis=1))
    #turn to continuous clusters label,from 0,1,2,3,...
    test_c={}
    for ind, i in enumerate(unique_labels):
        test_c[i]=ind
    y_pred=np.asarray([test_c[i] for i in y_pred],dtype=int)
    ##turn to categories
    labels=y_pred.astype('U')
    labels=pd.Categorical(values=labels,categories=natsorted(np.unique(y_pred).astype('U')))
    return labels

def presettings(save_dir="result_scRNA",dpi=200,verbosity=3):
    if not os.path.exists(save_dir):
        print("Warning:"+str(save_dir)+"does not exists, so we  will creat it automatically!!:\n")
        os.mkdir(save_dir)
    figure_dir=os.path.join(save_dir,"figures")
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    sc.settings.figdir=figure_dir
    sc.settings.verbosity=verbosity
    sc.settings.set_figure_params(dpi=dpi)
    sc.logging.print_versions()

def creatadata(datadir=None,exprmatrix=None,expermatrix_filename="matrix.mtx",is_mtx=True,cell_info=None,cell_info_filename="barcodes.tsv",gene_info=None,gene_info_filename="genes.tsv",project_name=None):
    """
    Construct a anndata object
    
    Construct a anndata from data in memory or files on disk. If datadir is a dir, there must be at least include "matrix.mtx" or data.txt(without anly columns name or rowname and sep="\t") , 

    """
    if (datadir is None and expermatrix is None and expermatrix_filename is None):
        raise ValueError("Please provide either the expression matrix or the ful path to the expression  matrix!!")
        #something wrong
    cell_info=pd.DataFrame(["cell_"+str(i) for i in range(1,x.shape[0]+1)],columns=["cellname"]) if cell_info is not None else cell_info
    gene_info=pd.DataFrame(["gene_"+str(i) for i in range(1,x.shape[1]+1)],columns=["genename"]) if gene_info is not None else gene_info 
    if datadir is not None:
        cell_and_gene_file = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
        if (os.path.isdir(datadir) and is_mtx==True): #sparse
            print("Start to read expression data (matrix.mtx)")
            x=sc.read_mtx(os.path.join(datadir,expermatrix_filename)).X.T
        else: #nonsparse
            x=pd.read_csv(os.path.join(datadir,expermatrix_filename),sep="\t",header=F)
       
            #only matrix with row names and colnames
        if cell_info_filename in cell_and_gene_file:
            cell_info=pd.read_csv(os.path.join(datadir,cell_info_filename),sep="\t",header=0,na_filter=False) 
        if gene_info_filename in cell_and_gene_file:
            gene_info=pd.read_csv(os.path.join(datadir,gene_info_filename),sep="\t",header=0,na_filter=False)
    else:
        x=exprmatrix # n*p matrix, cell* gene
  
    adata=sc.AnnData(x,obs=cell_info,var=gene_info)
    a=adata.obs["cellname"] if "cellname" in adata.obs.keys() else adata.obs.index
    adata.var_names=adata.var["genename"] if "genename" in adata.var.keys() else adata.var.index
    adata.obs_names_make_unique(join="-")
    adata.var_names_make_unique(join="-")
    adata.uns["ProjectName"]="DEC_clust_algorithm" if project_name is None else project_name 
    return adata

def prefilter_cells(adata,min_counts=None,max_counts=None,min_genes=200,max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[0],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_genes=min_genes)[0]) if min_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_genes=max_genes)[0]) if max_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw=sc.pp.log1p(adata,copy=True) #check the rowname 
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:",adata.raw.var_names.is_unique)
        
def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)

def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)
 
def normalize_log1p_scale(adata,units="UMI",n_top_genes=1000):
    if units=="UMI" or units== "CPM":
        sc.pp.normalize_per_cell(adata,counts_per_cell_after=10e4)
    sc.pp.filter_genes_dispersion(adata,n_top_genes=n_top_genes)
    sc.pp.log1p(adata)
    sc.pp.scale(adata,zero_center=True,max_value=6)
                    
#creat DEC object
def get_xinput(adata):
    if not isinstance(adata,AnnData):
        raise ValueError("adata must be an AnnData object")
    if issparse(adata.X):
        x=adata.X.toarray()
    else:
        x=adata.X
    return x

def getdims(dim):
    x=dim
    assert len(x)==2
    n_sample=x[0]
    if n_sample>20000:
        dims=[x[1],128,32]
    elif n_sample>10000:
        dims=[x[1],64,32]
    elif n_sample>5000:
        dims=[x[1],32,16]
    elif n_sample>2000:
        dims=[x[1],128]
    elif n_sample>500:
        dims=[x[1],64] #
    else:
        dims=[x[1],16] # or 32
    return dims

def OriginalClustering(adata,resolution=1.2,n_neighbors=20,n_comps=50,n_PC=20,n_job=4,dotsne=True,doumap=True,dolouvain=True):
    #Do PCA directly
    sc.tl.pca(adata,n_comps=n_comps)
    n_pcs=n_PC if n_PC<n_comps else n_comps
    #Do tsne based pca result
    if dotsne:
        sc.tl.tsne(adata,random_state=2,learning_rate=150,n_pcs=n_PC,n_jobs=n_job)
        #Save original X
        adata.obsm["X_tsne.ori"]=adata.obsm['X_tsne']
    #Do umap 
    if doumap:
        sc.pp.neighbors(adata,n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        #Save original
        adata.obsm['X_umap.ori']=adata.obsm['X_umap']
    if dolouvain:
        sc.tl.louvain(adata,resolution=resolution)
        adata.obs['louvain_ori']=adata.obs['louvain']
    print("OriginalClustering has completed!!!")

def first2prob(adata):
    first2ratio=[name for name in adata.uns.key() if str(name).startswith("prob_matrix")]
    for key_ in first2ratio:
        q_pred=adata.uns[key_]
        q_pred_sort=np.sort(q_pred,axis=1)
        y=q_pred_sort[:,-1]/q_pred_sort[:,-2]
        adata["first2ratio_"+str(key_).split("matrix")[1]]=y

def expand_grid(dictionary):
    from itertools import product
    return pd.DataFrame([row for row in product(*dictionary.values())],columns=dictionary.keys())
