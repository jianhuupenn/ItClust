from __future__ import division
import os

from . SAE import SAE  # load Stacked autoencoder
from . preprocessing import change_to_continuous
from time import time
import numpy as np
from keras.engine.topology import Layer, InputSpec
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import scanpy.api as sc
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
import keras.backend as K

class ClusteringLayer(Layer): # Re-define lot of build in functions for Keras
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clustering')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs): # The activation function for clustering layer
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DEC(object):
    def __init__(self,
                 dims,
                 x, # input matrix, row sample, col predictors 
                 y=None, # if provided will trained with supervised
                 alpha=1.0,
                 init='glorot_uniform', #initialization method
                 n_clusters=None,     # Number of Clusters, if provided, the clusters center will be initialized by K-means,
                 louvain_resolution=1.0, # resolution for louvain 
                 n_neighbors=10,    # the 
                 pretrain_epochs=200, # epoch for autoencoder
                 ae_weights=None, #ae_
                 actinlayer1="tanh",# activation for the last layer in encoder, and first layer in the decoder 
                 is_stacked=True,
                 transfer_feature=None,
                 model_weights=None,
                 y_trans=None,
                 softmax=False,
                 ):

        super(DEC, self).__init__()
        self.dims = dims
        self.x=x #feature n*p, n:number of cells, p: number of genes
        self.y=y # for supervised 
        self.y_trans=y_trans
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.is_stacked=is_stacked
        self.resolution=louvain_resolution
        self.alpha = alpha
        self.actinlayer1=actinlayer1
        self.transfer_feature=transfer_feature
        self.model_weights=model_weights
        self.softmax=softmax
        self.pretrain_epochs=pretrain_epochs
        if  self.transfer_feature is None:
            self.pretrain(n_neighbors=n_neighbors,epochs=self.pretrain_epochs,n_clusters=n_clusters)
        else:
            self.pretrain_transfer(n_neighbors=n_neighbors,model_weights=self.model_weights,features=transfer_feature,epochs=self.pretrain_epochs,n_clusters=n_clusters,y_trans=self.y_trans)

    def pretrain(self, optimizer='adam', epochs=200, n_neighbors=10,batch_size=256,n_clusters=None):
        print("Doing DEC: pretrain")  
        sae=SAE(dims=self.dims,drop_rate=0.2,batch_size=batch_size,actinlayer1=self.actinlayer1)# batch_size
        print('...Pretraining...')
        # begin pretraining
        t0 = time()
        if self.is_stacked:
            sae.fit(self.x,epochs=epochs)
        else:
            sae.fit2(self.x,epochs=epochs)

        self.autoencoder=sae.autoencoders
        self.encoder=sae.encoder
        print('Pretraining time: ', time() - t0)
        self.pretrained = True

        #build dec model and initialize model
        features=self.extract_features(self.x)
        features=np.asarray(features)
        if self.y is None: # Train data not labeled
            if isinstance(n_clusters,int): # Number of clusters known, use k-means
                print("...number of clusters have been specified, Initializing Cluster centroid  using K-Means")
                kmeans = KMeans(n_clusters=n_clusters, n_init=20)
                Y_pred_init = kmeans.fit_predict(features)
                self.init_pred= np.copy(Y_pred_init)
                self.n_clusters=n_clusters
                cluster_centers=kmeans.cluster_centers_
                self.init_centroid=cluster_centers
            else: # Number of clustered unknow, use unsupervised method
                print("...number of clusters does not know, Initialize Cluster centroid using louvain")
                adata=sc.AnnData(features)
                sc.pp.neighbors(adata, n_neighbors=n_neighbors)
                sc.tl.louvain(adata,resolution=self.resolution)
                Y_pred_init=adata.obs['louvain']
                self.init_pred=np.asarray(Y_pred_init,dtype=int)
                features=pd.DataFrame(features,index=np.arange(0,features.shape[0]))
                Group=pd.Series(self.init_pred,index=np.arange(0,features.shape[0]),name="Group")
                Mergefeature=pd.concat([features,Group],axis=1)
                cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
                self.n_clusters=cluster_centers.shape[0]
                self.init_centroid=cluster_centers
            print("The shape of cluster_centers",cluster_centers.shape)
        else: #  train data is labeled
            print("y known, initilize Cluster centroid using y")
            # build dec model
            features=pd.DataFrame(features,index=np.arange(0,features.shape[0]))
            Group=pd.Series(self.y.values,index=np.arange(0,features.shape[0]),name="Group")
            Mergefeature=pd.concat([features,Group],axis=1)
            cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
            self.n_clusters=cluster_centers.shape[0]
            self.init_centroid=cluster_centers
            print("The shape of cluster_center is",cluster_centers.shape)
        if not self.softmax:       # Use dec method to do clustering
            clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        else:        # Use softmax to do clustering
            clustering_layer=Dense(self.n_clusters,kernel_initializer="glorot_uniform",name="clustering",activation='softmax')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)



    def pretrain_transfer(self,features,model_weights,y_trans=None,optmizer="adam",n_neighbors=10,epochs=200,batch_size=32,n_clusters=None):
        #y_trans  is a numpy array
        print("Doing DEC: pretrain_transfer")
        if isinstance(n_clusters,int):
            print("...number of clusters have been specified, Initializing Cluster centroid  using K-Means")
            kmeans = KMeans(n_clusters=n_clusters, n_init=20)
            Y_pred_init = kmeans.fit_predict(features)
            self.init_pred= np.copy(Y_pred_init)
            self.n_clusters=n_clusters
            cluster_centers=kmeans.cluster_centers_
            self.init_centroid=[cluster_centers]
        else:
            print("The shape of features is",features.shape)
            if y_trans is not None and y_trans.shape[0]==features.shape[0]:
                print("The shape of y_trans is",y_trans.shape)
                print("...predicted y_test known, use it to get n_cliusters and init_centroid")
                self.init_pred=y_trans
                features=pd.DataFrame(features,index=np.arange(0,features.shape[0]))
                Group=pd.Series(y_trans,index=np.arange(0,features.shape[0]),name="Group")
                Mergefeature=pd.concat([features,Group],axis=1)
                cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
                self.n_clusters=cluster_centers.shape[0]
                self.init_centroid=cluster_centers
            else:
                print("...number of clusters does not know, Initialize Cluster centroid using louvain")
                #can be replaced by other clustering methods
                adata=sc.AnnData(features)
                sc.pp.neighbors(adata, n_neighbors=n_neighbors) #louvain step1
                sc.tl.louvain(adata,resolution=self.resolution) #louvain step2
                Y_pred_init=adata.obs['louvain']
                self.init_pred=np.asarray(Y_pred_init,dtype=int)
                features=pd.DataFrame(features,index=np.arange(0,features.shape[0]))
                Group=pd.Series(self.init_pred,index=np.arange(0,features.shape[0]),name="Group")
                Mergefeature=pd.concat([features,Group],axis=1)
                cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
                self.n_clusters=cluster_centers.shape[0]
                self.init_centroid=cluster_centers
                print("The shape of cluster_centers",cluster_centers.shape[0])

        sae=SAE(dims=self.dims,drop_rate=0.2,batch_size=batch_size,actinlayer1=self.actinlayer1)# batch_size
        self.autoencoder=sae.autoencoders
        self.encoder=sae.encoder
        clustering_layer=ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output) # use dec to do clustering
        self.model=Model(self.encoder.input,outputs=clustering_layer)
        print("The length layers  of self.model",len(self.model.layers))
        for i in range(len(self.model.layers)-2):
            self.model.layers[i+1].set_weights(model_weights[i+1])
        self.model.get_layer("clustering").set_weights([self.init_centroid])
        #fine tunning
    
    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def fit(self,x, maxiter=2e3, epochs_fit=10,batch_size=256, tol=1e-3): # unsupervised
        print("Doing DEC: fit")
        #step1 initial weights by louvain,or Kmeans
        self.model.get_layer(name='clustering').set_weights([self.init_centroid])
        y_pred_last = np.copy(self.init_pred)
        # Step 2: deep clustering
        # logging file
        #y_pred_last=self.init_pred
        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            q = self.model.predict(x, verbose=0)
            p = self.target_distribution(q)  # update the auxiliary target distribution p
            # evaluate the clustering performance
            y_pred = q.argmax(1)

             # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopped training.')
                break
            print("The value of delta_label of current",str(ite+1),"th iteration is",delta_label,">= tol",tol)
            #train on whole dataset on prespecified batch_size
            callbacks=[EarlyStopping(monitor='loss',min_delta=10e-4,patience=4,verbose=0,mode='auto')]
            self.model.fit(x=x,y=p,epochs=epochs_fit,batch_size=batch_size,callbacks=callbacks,shuffle=True,verbose=False)

        y0=pd.Series(y_pred)
        print("The final prediction cluster is:")
        print(y0.value_counts())
        Embeded_z=self.encoder.predict(x)
        return Embeded_z,q

    #Show the trajectory of the centroid during iterations
    def fit_trajectory(self,x, maxiter=2e3, epochs_fit=10,batch_size=256, tol=1e-3): # unsupervised
        print("Doing DEC: fit_trajectory")
        #step1 initial weights by louvain,or Kmeans
        self.model.get_layer(name='clustering').set_weights([self.init_centroid])
        y_pred_last = np.copy(self.init_pred)
        # Step 2: deep clustering
        # logging file
        #y_pred_last=self.init_pred
        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        trajectory_z=[] #trajectory embedding
        trajectory_l=[] #trajectory label
        for ite in range(int(maxiter)):
            q = self.model.predict(x, verbose=0)
            p = self.target_distribution(q)  # update the auxiliary target distribution p
            # evaluate the clustering performance
            y_pred = q.argmax(1)

             # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopped training.')
                break
            print("The value of delta_label of current",str(ite+1),"th iteration is",delta_label,">= tol",tol)
            #train on whole dataset on prespecified batch_size
            callbacks=[EarlyStopping(monitor='loss',min_delta=10e-4,patience=4,verbose=0,mode='auto')]
            self.model.fit(x=x,y=p,epochs=epochs_fit,batch_size=batch_size,callbacks=callbacks,shuffle=True,verbose=False)
            

            if ite % 10 ==0:
                print("This is the iteration of ", ite)
                Embeded_z=self.encoder.predict(x) # feature
                q_tmp=self.model.predict(x,verbose=0) # predicted clustering results
                l_tmp=change_to_continuous(q_tmp)
                trajectory_z.append(Embeded_z)
                trajectory_l.append(l_tmp)
        
        y0=pd.Series(y_pred)
        print("The final prediction cluster is:")
        print(y0.value_counts())
        Embeded_z=self.encoder.predict(x)
        return trajectory_z, trajectory_l,  Embeded_z, q
        #return ret


    def fit_supervise(self,x,y,epochs=2e3,batch_size=256):
        #y is 1-D array, Series, or a list, len(y)==x.shape[0]
        print("Doing DEC: fit_supervised")
        if self.softmax==False: # Only DEC clustering can set init_centroid
            self.model.get_layer(name='clustering').set_weights([self.init_centroid])
        y0=pd.Series(y,dtype="category") #avoding y is string
        y0=y0.cat.rename_categories(range(len(y0.cat.categories)))
        y_true=pd.get_dummies(pd.Series(y0)).values# coded according to 0,1,...,3
        y_true=y_true+0.00005*np.random.random(y_true.shape)+0.00001 # add some disturb
        y_true=y_true/y_true.sum(axis=1)[:,None]
        callbacks=[EarlyStopping(monitor='loss',min_delta=10e-4,patience=4,verbose=0,mode='auto')]
        self.model.fit(x=x,y=y_true,epochs=int(epochs),batch_size=batch_size,callbacks=callbacks,shuffle=True,verbose=False,validation_split=0.25)
        Embeded_z=self.encoder.predict(x) # feature
        q=self.model.predict(x,verbose=0) # predicted clustering results
        #return y0, representing the mapping reference for y
        return Embeded_z,q
