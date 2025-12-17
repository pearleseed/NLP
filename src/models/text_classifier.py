import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TextClassifier:

    def __init__(self, vectorizer: Any, model: Any = None):
        self.vectorizer = vectorizer
        self.model = model if model is not None else LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        self._is_fitted = False
    
    def fit(self, texts: List[str], labels: List[int]) -> 'TextClassifier':
        # Vectorize the texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train the model
        self.model.fit(X, labels)
        self._is_fitted = True
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)
    
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def get_feature_importance(self, top_n: int = 10) -> Optional[Dict[str, List]]:
        if not self._is_fitted:
            return None
        
        if not hasattr(self.model, 'coef_'):
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.model.coef_
        
        result = {}
        for i, class_coef in enumerate(coef):
            top_indices = np.argsort(class_coef)[-top_n:][::-1]
            result[i] = [(feature_names[idx], class_coef[idx]) for idx in top_indices]
        
        return result
