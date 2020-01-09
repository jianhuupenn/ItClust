from __future__ import division
from time import time
####
from . DEC import DEC
from . preprocessing import *
####
from keras.models import Model
import os,csv
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from scipy.sparse import issparse
import scanpy.api as sc
from anndata import AnnData
from natsort import natsorted 
from sklearn import cluster, datasets, mixture,metrics
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

class transfer_learning_clf(object):
    '''
    The transfer learning clustering and classification model. 
    This class has following api: fit(), predict(), Umap(), tSNE()
    '''
    def __init__(self):
        super(transfer_learning_clf, self).__init__()

    def fit(self,
            source_data, #adata
            target_data, #adata
            batch_size=256,
            maxiter=1000,
            pretrain_epochs=300,
            epochs_fit=5,
            tol=[0.001],
            alpha=[1.0],
            resolution=[0.2,0.4,0.8,1.2,1.6],
            n_neighbors=20,
            softmax=False,
            init="glorot_uniform",
            save_atr="isy_trans_True"
            ):
        ''' 
        Fit the transfer learning model using provided data.
        This function includes preprocessing steps.
        Input: source_data(anndata format), target_data(anndata format).
        Source and target data can be in any form (UMI or TPM or FPKM)
        Retrun: No return
        '''
        self.batch_size=batch_size
        self.maxiter=maxiter
        self.pretrain_epochs=pretrain_epochs
        self.epochs_fit=epochs_fit
        self.tol=tol
        self.alpha=alpha
        self.source_data=source_data
        self.target_data=target_data
        self.resolution=resolution
        self.n_neighbors=n_neighbors
        self.softmax=softmax
        self.init=init
        self.save_atr=save_atr
        dictionary={"alpha":alpha,"tol":tol,"resolution":resolution}
        df_expand=expand_grid(dictionary)
        #begin to conduct 
        adata_tmp=[]
        source_data.var_names_make_unique(join="-")
        source_data.obs_names_make_unique(join="-")

        #pre-processiong
        #1.pre filter cells
        prefilter_cells(source_data,min_genes=100) 
        #2 pre_filter genes
        prefilter_genes(source_data,min_cells=10) # avoiding all gene is zeros
        #3 prefilter_specialgene: MT and ERCC
        prefilter_specialgenes(source_data)
        #4 normalization,var.genes,log1p,scale
        sc.pp.normalize_per_cell(source_data)
        #5 scale
        sc.pp.log1p(source_data)
        sc.pp.scale(source_data,zero_center=True,max_value=6)
        source_data.var_names=[i.upper() for i in list(source_data.var_names)]#avoding some gene have lower letter
        adata_tmp.append(source_data) 

        #Target data
        target_data.var_names_make_unique(join="-")
        target_data.obs_names_make_unique(join="-")
        #pre-processiong
        #1.pre filter cells
        prefilter_cells(target_data,min_genes=100) 
        #2 pre_filter genes
        prefilter_genes(target_data,min_cells=10) # avoiding all gene is zeros
        #3 prefilter_specialgene: MT and ERCC
        prefilter_specialgenes(target_data)
        #4 normalization,var.genes,log1p,scale
        sc.pp.normalize_per_cell(target_data)

        # select top genes
        if target_data.X.shape[0]<=1500:
            ng=500
        elif 1500<target_data.X.shape[0]<=3000:
            ng=1000
        else:
            ng=2000

        sc.pp.filter_genes_dispersion(target_data, n_top_genes=ng)
        sc.pp.log1p(target_data)
        sc.pp.scale(target_data,zero_center=True,max_value=6)
        target_data.var_names=[i.upper() for i in list(target_data.var_names)]#avoding some gene have lower letter
        adata_tmp.append(target_data)
    
        #Concat *adata
        full_adata=AnnData.concatenate(*adata_tmp,join='inner',batch_key="dataset_batch",batch_categories=["source","target"]) #inner
        del adata_tmp
        del target_data
        del source_data
        ref_id=full_adata.obs["dataset_batch"]=="source"
        adata_test=full_adata[~ref_id,:].copy()
        adata_train=full_adata[ref_id,:].copy()
        if issparse(adata_train.X):
            x_train=adata_train.X.toarray()
        else:
            x_train=adata_train.X
        
        y_train=pd.Series(adata_train.obs["celltype"],dtype="category")
        y_train=y_train.cat.rename_categories(range(len(y_train.cat.categories)))
        print("The number of training celltypes is: ", len(set(y_train)))

        if issparse(adata_test.X):
            x_test=adata_test.X.toarray()
        else:
            x_test=adata_test.X

        #Training Data dec
        print("Training the source network")
        dims=getdims(x_train.shape)
        #dims=[x_train.shape[1],128,64]
        print("The layer numbers are"+str(dims[1:]))   
        print(":".join(["The shape of xtrain is",str(x_train.shape[0]),str(x_train.shape[1])]))
        print(":".join(["The shape of xtest is",str(x_test.shape[0]),str(x_test.shape[1])]))
        assert x_train.shape[1]==x_test.shape[1]
        dec=DEC(dims=dims,y=y_train,x=x_train,alpha=alpha,init=self.init,pretrain_epochs=self.pretrain_epochs,actinlayer1="tanh",softmax=softmax)
        dec.compile(optimizer=SGD(lr=0.01,momentum=0.9))
        #print("dec.init_centroid",type(dec.init_centroid),dec.init_centroid)
        Embeded_z,q_pred=dec.fit_supervise(x=x_train,y=y_train,epochs=2e3,batch_size=self.batch_size) # fine tunning
    
        #---------------------------------------------------------------------------------------------------
        weights=[i0.get_weights() for i0 in dec.model.layers]
        features=dec.encoder.predict(x_test)
        q=dec.model.predict(x_test,verbose=0)

        #np.savetxt("testq.txt",q)
        print("Training model finished! Start to fit target network!")
        val_y_pre=dec.model.predict(x_train,verbose=0)
        val_y_pre=[np.argmax(i) for i in val_y_pre]
        val_ari=metrics.adjusted_rand_score(val_y_pre,y_train.tolist())
        t0=time()
        dec2=DEC(dims=dims,x=x_test,alpha=alpha,init=self.init,pretrain_epochs=self.pretrain_epochs,actinlayer1="tanh",softmax=softmax,transfer_feature=features,model_weights=weights,y_trans=q.argmax(axis=1))
        dec2.compile(optimizer=SGD(0.01,0.9))
        trajectory_z, trajectory_l, Embeded_z,q_pred=dec2.fit_trajectory(x=x_test,tol=tol,epochs_fit=self.epochs_fit,batch_size=self.batch_size)# Fine tunning
        print("How many trajectories ", len(trajectory_z))
        for i in range(len(trajectory_z)):
            adata_test.obsm["trajectory_Embeded_z_"+str(i)]=trajectory_z[i]
            adata_test.obs["trajectory_"+str(i)]=trajectory_l[i]

        labels=change_to_continuous(q_pred)
        adata_test.obsm["X_Embeded_z"+str(self.save_atr)]=Embeded_z
        adata_test.obs["dec"+str(self.save_atr)]=labels
        adata_test.obs["maxprob"+str(self.save_atr)]=q_pred.max(1)
        adata_test.obsm["prob_matrix"+str(self.save_atr)]=q_pred
        adata_test.obsm["X_pcaZ"+str(self.save_atr)]=sc.tl.pca(Embeded_z)
        
        self.adata_train=adata_train
        self.adata_test=adata_test
        self.dec2=dec2
        self.labels=labels

    def predict(self,save_dir="./results", write=True):
        ''' 
        Will return clustering prediction(DataFrame), 
        clustering probability (DataFrame) and 
        celltype assignment confidence score(dictionary).
        If write is True(default), results will also be written provided save_dir
        '''
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        # Cluster prediction 
        pred = {'cell_id': self.adata_test.obs.index.tolist(), 'cluster': self.adata_test.obs["decisy_trans_True"].tolist()}
        pred = pd.DataFrame(data=pred)
        # Confidence score
        celltype_pred={}
        source_label=pd.Series(self.adata_train.obs["celltype"],dtype="category")
        source_label=source_label.cat.categories.tolist()
        target_label=self.adata_test.obs["decisy_trans_True"].cat.categories.tolist()
        for i in range(len(target_label)):
            end_cell=self.adata_test.obs.index[self.adata_test.obs["decisy_trans_True"]==target_label[i]]
            start_cell=self.adata_test.obs.index[self.adata_test.obs["trajectory_0"]==target_label[i]]
            overlap=len(set(end_cell).intersection(set(start_cell)))
            celltype_pred[target_label[i]]=[source_label[i],round(overlap/len(end_cell),3)]

        # Clustering probability 
        prob=pd.DataFrame(self.adata_test.obsm["prob_matrix"+str(self.save_atr)])
        prob.index=self.adata_test.obs.index.tolist()
        prob.columns=["cluster"+str(i) for i in range(len(set(prob.columns)))]
        if write:
            pred.to_csv(save_dir+"/clustering_results.csv")
            prob.to_csv(save_dir+"/clustering_prob.csv")
            f=open(save_dir+"/celltype_assignment.txt","w")
            for k, v in celltype_pred.items():
                f.write("Cluster "+str(k)+" is "+str(v[1]*100)+"%"+" to be "+v[0]+" cell\n")
        
            f.close()
            print("Results are written to ",save_dir)

        return pred, prob, celltype_pred

    def Umap(self):
        '''
        Do Umap.
        Return: the umap projection(DataFrame)
        '''
        print("Doing U-map!")
        sc.pp.neighbors(self.adata_test, n_neighbors=10,use_rep="X_Embeded_z"+str(self.save_atr))
        sc.tl.umap(self.adata_test)
        self.adata_test.obsm['X_umap'+str(self.save_atr)]=self.adata_test.obsm['X_umap'].copy()
        return self.adata_test.obsm['X_umap'+str(self.save_atr)]

    def tSNE(self):
        '''
        Do tSNE.
        Return: the umap projection(DataFrame)
        '''
        print("Doing t-SNE!")
        sc.tl.tsne(self.adata_test,use_rep="X_Embeded_z"+str(self.save_atr),learning_rate=150,n_jobs=10)
        self.adata_test.obsm['X_tsne'+(self.save_atr)]=self.adata_test.obsm['X_tsne']
        return self.adata_test.obsm['X_tsne'+(self.save_atr)]





