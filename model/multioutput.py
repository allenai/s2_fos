import numpy as np
from sklearn.multioutput import MultiOutputClassifier

# linearSVC has no predict_proba and MultiOutputClassifier has no decision_function
# so we need this little wrapper
class MultiOutputClassifierWithDecision(MultiOutputClassifier):
    def decision_function(self, X):
        results = [estimator.decision_function(X) for estimator in self.estimators_]
        return np.array(results).squeeze().T  # num_examples X num_classes
