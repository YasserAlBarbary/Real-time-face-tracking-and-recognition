import os
import pickle
import numpy as np
from sklearn import neighbors, svm
#import sklearn.svm.SVC as svc
BASE_DIR = os.path.dirname(__file__) + '/'
PATH_TO_PKL = 'trained_classifier.pkl'


class FaceClassifier:
    def __init__(self, model_path=None):
        self.learning_rate = 0.001
        self.training_epochs = 20
        self.n_inputs = 128
        
        self.hidden_layer_params= 256
        
        
        self.model = None
        if model_path is None:
            return
        elif model_path == 'default':
            model_path = BASE_DIR+PATH_TO_PKL

        # Load models
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def train(self, X, y, model='knn', save_model_path=None):
        if model == 'knn':
            self.model = neighbors.KNeighborsClassifier(3, weights='uniform')
        else:  # svm
            self.model = svm.SVC(kernel='rbf', probability=True)
        self.model.fit(X, y)
        if save_model_path is not None:
            with open(save_model_path, 'wb') as f:
                pickle.dump(self.model, f)

    def classify(self, descriptor):
        if self.model is None:
            print('Train the model before doing classifications.')
            return
        #print("+++++++++++++++++++")
        #print(descriptor.shape)
        predictions = self.model.predict_proba(descriptor)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
#        print("preditions ",predictions[0])
#        print("best indx",best_class_indices[0])
#        print("bes prob %0.2f" %( best_class_probabilities[0]))
#        print("-------------------")
        name = self.model.predict([descriptor.squeeze()])[0]
#        print(name)
        conf = False
        if 2*(1/len(predictions[0] ))< best_class_probabilities[0] : #threshold for confidence
            conf =True
        return (self.model.predict([descriptor.squeeze()])[0] ,str( round(best_class_probabilities[0]*100,2) ) ,conf ) 
