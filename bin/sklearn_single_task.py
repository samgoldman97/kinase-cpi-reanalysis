""" 

This file defines generic sklearn models to be used with the protein and
compound interaction data, such that inside the model fit, many different
regressors are trained treating each protein and each compound as its own
respective case. 

"""


from math import ceil
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.neural_network import MLPRegressor
from mlp_ensemble import MLPEnsembleRegressor
from hybrid  import HybridMLPEnsembleGP

from utils import tprint
from tqdm import tqdm


PROT_FEATURE = 3706
CHEM_FEATURE = 28

class SingleTaskModel(object):

    def __init__(self, normalize = False):

        # Define scalers
        self.normalize = normalize
        if self.normalize: 
            self.scaler = False

        self.models_ =  []
        self.model_dict = ()

    def remove_duplicates(self, X): 
        return np.vstack({tuple(row) for row in X})

    def fit_single_model(self, X,y): 
        """ Abstract method to return a model that fits to X and y"""
        raise NotImplementedError()

    def fit(self, X, y):

        if self.normalize:
            self.scaler = StandardScaler()
            y = self.scaler.fit_transform(y.reshape(-1,1))

        # Use these for the novel examples later
        self.prot_to_preds = defaultdict(lambda : []) 
        self.chem_to_preds = defaultdict(lambda : []) 

        prot_feats = X[:, - PROT_FEATURE: ]
        chem_feats = X[:, : -PROT_FEATURE]  

        # Verify size 
        assert (prot_feats.shape[1] == PROT_FEATURE)

        # Unique proteins and chems
        unique_prot = self.remove_duplicates(prot_feats)
        unique_chem = self.remove_duplicates(chem_feats)

        print(f"Num unique proteins: {len(unique_prot)}")
        print(f"Num unique mols: {len(unique_chem)}")
        self.prot_task_models = defaultdict(lambda : None)
        self.chem_task_models = defaultdict(lambda : None) 

        self.prot_task_models.update({tuple(j) : None 
                                      for j in unique_prot})
        self.chem_task_models.update({tuple(j) : None
                                      for j in unique_chem})
        
        # Now train each prot task model 
        print("Training prot task models")
        for index, prot_task in tqdm(enumerate(self.prot_task_models.keys())):  
            # Get all indices
            prot_task_ar = np.array(prot_task)
            truth_mask = np.all(prot_feats == prot_task_ar, axis = 1)
            X_temp, y_temp = chem_feats[truth_mask], y[truth_mask]
            new_model = self.fit_single_model(X_temp, y_temp) 
            self.prot_task_models[prot_task] = new_model


        # Now train each chem task model 
        print("Training chem task models")
        for chem_task in tqdm(self.chem_task_models.keys()):  
            # Get all indices
            chem_task_ar = np.array(chem_task)
            truth_mask = np.all(chem_feats  == chem_task_ar, axis = 1)
            X_temp, y_temp = prot_feats[truth_mask], y[truth_mask]

            new_model = self.fit_single_model(X_temp, y_temp) 
            self.chem_task_models[chem_task] = new_model

        return self

    def predict(self, X):
        """ Predict """

        prot_feats = X[:, - PROT_FEATURE: ]
        chem_feats = X[:, : -PROT_FEATURE]  

        # Verify size 
        assert (prot_feats.shape[1] == PROT_FEATURE)

        # Unique proteins and chems
        unique_prot = self.remove_duplicates(prot_feats)
        unique_chem = self.remove_duplicates(chem_feats)

        chem_task_to_pred = defaultdict(lambda : [])
        prot_task_to_pred = defaultdict(lambda : [])
        new_quadrant_points = []
        for ind, (prot_feat, chem_feat) in enumerate(zip(prot_feats, chem_feats)): 
            prot_feat_tup = tuple(prot_feat)
            chem_feat_tup = tuple(chem_feat)

            chem_task_model_exists = chem_feat_tup in self.chem_task_models
            prot_task_model_exists = prot_feat_tup in self.prot_task_models

            if chem_task_model_exists:
                chem_task_to_pred[chem_feat_tup].append(ind)

            if prot_task_model_exists:
                prot_task_to_pred[prot_feat_tup].append(ind)

            if (not prot_task_model_exists) and (not chem_task_model_exists):
                new_quadrant_points.append((prot_feat_tup, 
                                            chem_feat_tup, ind))

        indices_preds = defaultdict(lambda : [])

        # Predict new proteins 
        print("Predicting new protein quadrant")
        for chem_task, indices in tqdm(chem_task_to_pred.items()): 
            model = self.chem_task_models[chem_task]
            X_prot_vals = prot_feats[indices]

            preds = model.predict(X_prot_vals)
            for pred, index in zip(preds, indices):
                prot_tuple = tuple(prot_feats[index])
                self.prot_to_preds[prot_tuple].append(pred)
                indices_preds[index].append(pred) 

        # Predict new chem
        print("Predicting new chem quadrant")
        for prot_task, indices in tqdm(prot_task_to_pred.items()): 
            model = self.prot_task_models[prot_task]

            X_chem_vals = chem_feats[indices]
            preds = model.predict(X_chem_vals)

            for pred, index in zip(preds, indices):
                chem_tuple = tuple(chem_feats[index])
                self.chem_to_preds[chem_tuple].append(pred)
                indices_preds[index].append(pred) 

        # Now predict new quadrant with simple, naive means
        print("Predicting novel quadrant")
        for prot_feat_tup, chem_feat_tup, index in tqdm(new_quadrant_points):
            mean_chem = np.mean(self.chem_to_preds[chem_feat_tup])
            mean_prot = np.mean(self.prot_to_preds[prot_feat_tup])
            indices_preds[index].extend([mean_chem, mean_prot])

        pred = np.zeros((1, X.shape[0]))

        for index in range(X.shape[0]): 
            pred_val = np.mean(indices_preds[index])
            pred[0, index] = pred_val 


        assert(pred.shape[1] == X.shape[0])

        # Expand sklearn preds
        if len(pred.shape) == 2: 
            pred = pred.reshape(*pred.shape, 1)

        assert(pred.shape[2] == 1)

        self.uncertainties_ = pred.var(0).flatten()
        self.multi_predict_ = pred[:, :, 0].T
        flattened = pred.mean(0).flatten()
        if self.normalize: 
            flattened = self.scaler.inverse_transform(
                flattened.reshape(-1,1).flatten()
            )

        return flattened

class LinearSingle(SingleTaskModel):

    def __init__(self, **kwargs):

        super(LinearSingle, self).__init__(**kwargs)

    def fit_single_model(self, X,y): 
        """ Make linear regresssion model """

        new_model = Ridge(
            alpha = 10,
            fit_intercept=True,
            normalize=True,
            copy_X=True
        )
        new_model.fit(X,y)
        return new_model

class GPSingle(SingleTaskModel):

    def __init__(self, kernel=None, 
                 n_restarts = 0,**kwargs):

        super(GPSingle, self).__init__(**kwargs)
        self.kernel_ = kernel
        self.n_restarts_ = n_restarts

    def fit_single_model(self, X,y): 
        """ Make linear regresssion model """
        new_model = GaussianProcessRegressor(
            kernel=self.kernel_,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_,
            copy_X_train=False,
        )

        new_model.fit(X,y)
        return new_model

class HybridSingle(SingleTaskModel):
    def __init__(self, kernel=None, 
                 n_restarts = 0,
                 norm_mlp = False,
                 mlp_backend = "keras",
                 **kwargs):
        super(HybridSingle, self).__init__(**kwargs)
        self.kernel_ = kernel
        self.n_restarts_ = n_restarts
        self.norm_mlp = norm_mlp
        self.backend_ = mlp_backend

    def fit_single_model(self, X,y): 
        """ Make linear regresssion model """
        from gaussian_process import GPRegressor
        gp_model = GPRegressor(
            backend='sklearn',#'gpytorch',
            n_restarts=10,
            verbose=True
        )

        # Simple sklearn implementation
        #mlp_regr = MLPRegressor(hidden_layer_sizes=[200,200],
        #                            activation='relu',
        #                            solver='adam',
        #                            alpha=0.1,
        #                            batch_size=500,
        #                            max_iter=50,
        #                            momentum=0.9,
        #                            nesterovs_momentum=True,
        #                            verbose=True
        #)

        mlp_regr = MLPEnsembleRegressor(
            layer_sizes_list=[[200,200]],
            activations='relu',
            solvers='adam',
            alphas=0.1,
            batch_sizes=500,
            split=False,
            max_iters=50,
            momentums=0.9,
            normalize = self.norm_mlp,
            backend=self.backend_,
            nesterovs_momentums=True,
            verbose=True
        )

        new_model =  HybridMLPEnsembleGP(mlp_regr, gp_model, 
                                         use_uq = False, 
                                         predict_flat = True)
        new_model.fit(X,y)
        return new_model
