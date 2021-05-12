import numpy as np

class HybridMLPEnsembleGP(object):
    def __init__(self, mlp_ensemble, gaussian_process, use_uq = True,
                 predict_flat = False):
        self.mlp_ensemble_ = mlp_ensemble
        self.gaussian_process_ = gaussian_process
        self.use_uq = use_uq
        self.predict_flat = predict_flat

    def fit(self, X, y):
        self.mlp_ensemble_.fit(X, y)

        y_pred = self.mlp_ensemble_.predict(X)

        # If using sklearn model, we need to reshape
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        self.gaussian_process_.fit(X, y - y_pred)

        #X_tiled = np.tile(X, (self.mlp_ensemble_.n_regressors_, 1))
        #y_tiled = np.tile(y.flatten(), self.mlp_ensemble_.n_regressors_)
        #y_pred_tiled = self.mlp_ensemble_.multi_predict_.flatten('F')
        #self.gaussian_process_.fit(X_tiled, y_tiled - y_pred_tiled)

    def predict(self, X):
        residual = self.gaussian_process_.predict(X)

        if self.use_uq: 
            self.uncertainties_ = self.gaussian_process_.uncertainties_

        prediction = self.mlp_ensemble_.predict(X) + residual
        if self.predict_flat:
            prediction = prediction.flatten()

        return prediction
