from math import ceil
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.activations import softplus
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from utils import tprint
from tqdm import tqdm

def check_param_length(param, n_regressors):
    if len(param) != n_regressors:
        raise ValueError('Invalid parameter list length')

def gaussian_nll(ytrue, ypreds):
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]

    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)

    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)


PROT_FEATURE = 3706
CHEM_FEATURE = 28

class MLPEnsembleRegressor(object):
    def __init__(self,
                 layer_sizes_list,
                 activations='relu',
                 loss='mse',
                 solvers='adam',
                 alphas=0.0001,
                 batch_sizes=None,
                 max_iters=200,
                 momentums=0.9,
                 split=False,
                 nesterovs_momentums=True,
                 backend='keras',
                 normalize=False,
                 random_state=1,
                 verbose=False,
    ):
        if random_state is not None:
            tf.set_random_seed(random_state)

        self.backend_ = backend
        self.verbose_ = verbose

        self.n_regressors_ = len(layer_sizes_list)
        self.layer_sizes_list_ = layer_sizes_list


        self.loss_ = loss

        # Activation functions.
        if issubclass(type(activations), list):
            check_param_length(activations, self.n_regressors_)
            self.activations_ = activations
        else:
            self.activations_ = [ activations ] * self.n_regressors_

        # Solvers.
        if issubclass(type(solvers), list):
            check_param_length(solvers, self.n_regressors_)
            self.solvers_ = solvers_
        else:
            self.solvers_ = [ solvers ] * self.n_regressors_

        # Alphas.
        if issubclass(type(alphas), list):
            check_param_length(alphas, self.n_regressors_)
            self.alphas_ = alphas_
        else:
            self.alphas_ = [ alphas ] * self.n_regressors_

        # Batch Sizes.
        if issubclass(type(batch_sizes), list):
            check_param_length(batch_sizes, self.n_regressors_)
            self.batch_sizes_ = batch_sizes_
        else:
            self.batch_sizes_ = [ batch_sizes ] * self.n_regressors_

        # Maximum number of iterations.
        if issubclass(type(max_iters), list):
            check_param_length(max_iters, self.n_regressors_)
            self.max_iters_ = max_iters
        else:
            self.max_iters_ = [ max_iters ] * self.n_regressors_

        # Momentums.
        if issubclass(type(momentums), list):
            check_param_length(momentums, self.n_regressors_)
            self.momentums_ = momentums_
        else:
            self.momentums_ = [ momentums ] * self.n_regressors_

        # Whether to use Nesterov's momentum.
        if issubclass(type(nesterovs_momentums), list):
            check_param_length(nesterovs_momentums, self.n_regressors_)
            self.nesterovs_momentums_ = nesterovs_momentums_
        else:
            self.nesterovs_momentums_ = [ nesterovs_momentums ] * self.n_regressors_

        # Define scalers
        self.normalize = normalize
        if self.normalize: 
            self.scaler = False

        # If true, treat each compound and protein as separate
        self.split = split
        if self.split: 
            self.models_ =  []
            if self.loss_ == 'gaussian_nll':
                raise NotImplementedError()

            if self.n_regressors_ != 1:
                raise NotImplementedError()

            #if self.backend_ != "keras":
            #    raise NotImplementedError()
            self.model_dict = ()


    def _create_models(self, X, y, clear = True):
        if len(y.shape) == 1:
            n_outputs = 1
        else:
            raise ValueError('Only scalar predictions are currently supported.')

        if clear: 
            self.models_ = []

        if self.backend_ == 'sklearn':
            from sklearn.neural_network import MLPRegressor

            for model_idx in range(self.n_regressors_):
                model = MLPRegressor(
                    hidden_layer_sizes=self.layer_sizes_list_[model_idx],
                    activation=self.activations_[model_idx],
                    solver=self.solvers_[model_idx],
                    alpha=self.alphas_[model_idx],
                    batch_size=self.batch_sizes_[model_idx],
                    max_iter=self.max_iters_[model_idx],
                    momentum=self.momentums_[model_idx],
                    nesterovs_momentum=self.nesterovs_momentums_[model_idx],
                    verbose=self.verbose_,
                )
                self.models_.append(model)

        elif self.backend_ == 'keras':
            from keras import regularizers
            from keras.layers import Dense
            from keras.models import Sequential

            for model_idx in range(self.n_regressors_):
                hidden_layer_sizes = self.layer_sizes_list_[model_idx]

                model = Sequential()
                for layer_size in hidden_layer_sizes:
                    model.add(Dense(layer_size, kernel_initializer='normal',
                                    activation=self.activations_[model_idx],
                                    kernel_regularizer=regularizers.l2(0.01)))

                if self.loss_ == 'mse':
                    model.add(
                        Dense(1, kernel_initializer='normal',
                              kernel_regularizer=regularizers.l2(0.01))
                    )
                    model.compile(loss='mean_squared_error',
                                  optimizer=self.solvers_[model_idx])

                elif self.loss_ == 'gaussian_nll':
                    model.add(
                        Dense(2, kernel_initializer='normal',
                              kernel_regularizer=regularizers.l2(0.01))
                    )
                    model.compile(loss=gaussian_nll,
                                  optimizer=self.solvers_[model_idx])

                self.models_.append(model)

    def remove_duplicates(self, X): 
        return np.vstack({tuple(row) for row in X})

    def fit(self, X, y):
        y = y.flatten()

        if self.normalize: 
            self.scaler = StandardScaler()
            y = self.scaler.fit_transform(y.reshape(-1,1)).flatten()

        if len(y) != X.shape[0]:
            raise ValueError('Data has {} samples and {} labels.'
                             .format(X.shape[0], len(y)))

        if self.verbose_:
            tprint('Fitting MLP ensemble with {} regressors'
                   .format(self.n_regressors_))

        # Splitting
        if self.split:  

            # Use these for the novel examples later
            self.prot_to_preds = defaultdict(lambda : []) 
            self.chem_to_preds = defaultdict(lambda : []) 

            prot_feats = X[:, -PROT_FEATURE: ]
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

                self._create_models(X_temp, y_temp, clear=False)
                self.prot_task_models[prot_task] = self.models_[-1]
                cur_model = self.prot_task_models[prot_task] 
                cur_model.fit(X_temp, y_temp)
                              #, batch_size=self.batch_sizes_[0], 
                              #epochs=self.max_iters_[0], 
                              #verbose=self.verbose_)

            # Now train each chem task model 
            print("Training chem task models")
            for chem_task in tqdm(self.chem_task_models.keys()):  
                # Get all indices
                chem_task_ar = np.array(chem_task)
                truth_mask = np.all(chem_feats  == chem_task_ar, axis = 1)
                X_temp, y_temp = prot_feats[truth_mask], y[truth_mask]

                self._create_models(X_temp, y_temp)
                self.chem_task_models[chem_task] = self.models_[-1]
                cur_model = self.chem_task_models[chem_task] 
                cur_model.fit(X_temp, y_temp)
                              #, batch_size=self.batch_sizes_[0], 
                              #epochs=self.max_iters_[0], 
                              #verbose=self.verbose_)
        else: 
            self._create_models(X, y)

            if self.backend_== 'sklearn':
                [ model.fit(X, y) for model in self.models_ ]

            elif self.backend_ == 'keras':
                [ model.fit(X, y,
                            batch_size=self.batch_sizes_[model_idx],
                            epochs=self.max_iters_[model_idx],
                            verbose=self.verbose_)
                  for model_idx, model in enumerate(self.models_) ]

            if self.verbose_:
                tprint('Done fitting MLP ensemble.')

        return self

    def predict(self, X):

        if self.split: 
            prot_feats = X[:, -PROT_FEATURE: ]
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

        else:
            pred = np.array([ model.predict(X) for model in self.models_ ])

        assert(pred.shape[0] == self.n_regressors_)
        assert(pred.shape[1] == X.shape[0])

        if self.loss_ == 'gaussian_nll':
            assert(pred.shape[2] == 2)

            pred_mean = pred[:, :, 0]
            pred_var = np.exp(pred[:, :, 1])

            ys = pred_mean.mean(0)
            self.uncertainties_ = (
                pred_var + np.power(pred_mean, 2)
            ).mean(0) - np.power(ys, 2)
            self.multi_predict_ = pred_mean.T

            if self.normalize: 
                raise NotImplementedError("Normalization not implemented for Gaussian NLL loss")

            if self.split: 
                raise NotImplementedError("Split implementation not implemented for Gaussian NLL loss")

            return ys


        elif self.loss_ == 'mse':
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

