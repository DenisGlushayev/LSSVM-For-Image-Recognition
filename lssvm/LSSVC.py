import numpy as np
from numpy import dot

from utils.kernel import get_kernel
from utils.import_export import dump_model, load_model
from utils.conversion import numpy_json_encoder


class LSSVC():
    
    def __init__(self, gamma=1, kernel='rbf', **kernel_params): 
        # Hyperparameters
        self.gamma = gamma
        self.kernel_ = kernel
        self.kernel_params = kernel_params
        
        # Model parameters
        self.alpha = None
        self.b = None
        self.sv_x = None
        self.sv_y = None
        self.y_labels = None
        
        self.K = get_kernel(kernel, **kernel_params)
    
    def _optimize_parameters(self, X, y_values):
        """Help function that optimizes the dual variables through the 
        use of the kernel matrix pseudo-inverse.
        """
        sigma = np.multiply(y_values*y_values.T, self.K(X,X))
        
        A = np.block([
            [0, y_values.T],
            [y_values, sigma + self.gamma**-1 * np.eye(len(y_values))]
        ])
        B = np.array([0]+[1]*len(y_values))
        
        A_cross = np.linalg.pinv(A)

        solution = dot(A_cross, B)
        b = solution[0]
        alpha = solution[1:]
        
        return (b, alpha)
    
    def fit_batch(self, X_batch, y_batch):
        y_reshaped = y_batch.reshape(-1, 1) if y_batch.ndim == 1 else y_batch

        if self.sv_x is None:
            # If this is the first batch, initialize the model
            self.sv_x = X_batch
            self.sv_y = y_reshaped
        else:
            # Append the current batch to the existing dataset
            if self.sv_x.shape[0] == 0:
                # If the dataset is empty, initialize it with the current batch
                self.sv_x = X_batch
                self.sv_y = y_reshaped
            else:
                self.sv_x = np.vstack((self.sv_x, X_batch))
                self.sv_y = np.vstack((self.sv_y, y_reshaped))

        self.y_labels = np.unique(self.sv_y, axis=0)

        if len(self.y_labels) == 2:
            # Binary classification
            y_values = np.where((self.sv_y == self.y_labels[0]).all(axis=1), -1, +1)[:, np.newaxis]
            self.b, self.alpha = self._optimize_parameters(self.sv_x, y_values)
        else:
            # Multiclass classification (one-vs-all approach)
            n_classes = len(self.y_labels)
            if self.b is None:
                self.b = np.zeros(n_classes)
                self.alpha = np.zeros((n_classes, len(self.sv_y)))
            else:
                # Update existing values incrementally
                for i in range(n_classes):
                    y_values = np.where((self.sv_y == self.y_labels[i]).all(axis=1), +1, -1)[:, np.newaxis]
                    self.b[i], self.alpha[i] = self._optimize_parameters(self.sv_x, y_values)
        
    def predict(self, X):
        """Predicts the labels of data X given a trained model.
        - X: ndarray of shape (n_samples, n_attributes)
        """
        if self.alpha is None:
            raise Exception(
                "The model doesn't see to be fitted, try running .fit() method first"
            )

        X_reshaped = X.reshape(1,-1) if X.ndim==1 else X
        KxX = self.K(self.sv_x, X_reshaped)
        
        if len(self.y_labels)==2: # binary classification
            y_values = np.where(
                (self.sv_y == self.y_labels[0]).all(axis=1),
                -1,+1)[:,np.newaxis]

            y = np.sign(dot(np.multiply(self.alpha, y_values.flatten()), KxX) + self.b)
            
            y_pred_labels = np.where(y==-1, self.y_labels[0], self.y_labels[1])
        
        else: # multiclass classification, one-vs-all approach
            y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where(
                    (self.sv_y == self.y_labels[i]).all(axis=1),
                    +1, -1)[:,np.newaxis]
                y[i] = dot(np.multiply(self.alpha[i], y_values.flatten()), KxX) + self.b[i]
            
            predictions = np.argmax(y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
            
        return y_pred_labels

    def dump(self, filepath='model', only_hyperparams=False):
        """This method saves the model in a JSON format.
        - filepath: string, default = 'model'
            File path to save the model's json.
        - only_hyperparams: boolean, default = False
            To either save only the model's hyperparameters or not, it 
            only affects trained/fitted models.
        """
        model_json = {
            'type': 'LSSVC',
            'hyperparameters': {
                'gamma': self.gamma,
                'kernel': self.kernel_,
                'kernel_params': self.kernel_params
            }           
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json['parameters'] = {
                'alpha': self.alpha,
                'b': self.b,
                'sv_x': self.sv_x,
                'sv_y': self.sv_y,
                'y_labels': self.y_labels
            }
        
        dump_model(model_dict=model_json, file_encoder=numpy_json_encoder, filepath=filepath)
        
    @classmethod
    def load(cls, filepath, only_hyperparams=False):
        """This class method loads a model from a .json file.
        - filepath: string
            The model's .json file path.
        - only_hyperparams: boolean, default = False
            To either load only the model's hyperparameters or not, it 
            only has effects when the dump of the model as done with the
            model's parameters.
        """
        model_json = load_model(filepath=filepath)

        if model_json['type'] != 'LSSVC':
            raise Exception(
                f"Model type '{model_json['type']}' doesn't match 'LSSVC'"
            )

        lssvc = LSSVC(
            gamma = model_json['hyperparameters']['gamma'],
            kernel = model_json['hyperparameters']['kernel'],
            **model_json['hyperparameters']['kernel_params']
        )

        if (model_json.get('parameters') is not None) and (not only_hyperparams):
            lssvc.alpha = np.array(model_json['parameters']['alpha'])
            lssvc.b = np.array(model_json['parameters']['b'])
            lssvc.sv_x = np.array(model_json['parameters']['sv_x'])
            lssvc.sv_y = np.array(model_json['parameters']['sv_y'])
            lssvc.y_labels = np.array(model_json['parameters']['y_labels'])

        return lssvc
        
