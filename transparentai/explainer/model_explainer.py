import shap
import warnings
import matplotlib.pyplot as plt
# import seaborn as sns


# warnings.filterwarnings('ignore')

class ModelExplainer():
    """
    Class that allows to understand a local prediction or model behavior using shap_ package.
    For the moment this class will work only for model that TreeExplainer or LinearExplainer of
    shap_ package can handle.
    
    .. _shap : https://github.com/slundberg/shap/
    """

    def __init__(self, model, X=None, feature_names=None, model_type='tree'):
        """
        Parameters
        ----------
        model:
            model to inspect
        X: pd.DataFrame (default, None)
            data (possibly training) to start the explainer
            mandatory for a linear model but for tree model it can improve computing time
            so it's recommanded to use it.
        feature_names: list (default, None)
            list of feature names (lenght == lenght of X columns)
        model_type: str (default, 'tree')
            Type of model to inspect, it can only be 'tree' or 'linear'
            
        Raises
        ------
        """
        if model_type not in ['tree','linear']:
            raise ValueError('model_type has to be one of the following : tree or linear')
        
        self.model_type = model_type
        
        if model_type == 'tree':
            if X is None:
                self.explainer = shap.TreeExplainer(model)
            else:
                if len(X) > 100:
                    # reduce X size to compute quicker
                    X = shap.sample(X, 100)
                    
                self.explainer = shap.TreeExplainer(model, X)
        elif model_type == 'linear':
            if X is None:
                raise ValueError('X should be set as data for the explainer')
            self.explainer = shap.LinearExplainer(model, X)
            
        self.feature_names = feature_names
        
    
    def shap_values(self, X):
        """
        Compute shap values for a given X. X can be just a row or more.
        
        Parameters
        ----------
        X: pd.DataFrame or np.array
            Data to explain
            
        Returns
        -------
        list:
            list of values returned by shap_values() function
        """
        if (self.feature_names is None) and (type(X) == type(pd.DataFrame())):
            self.feature_names = X.columns
        
        values = self.explainer.shap_values(X)
        
        if self.model_type == 'tree':
            values = values[1]
            
        return values
        
    def explain_local(self, X):
        """
        Explain a local prediction : only one row required.
        
        Parameters
        ----------
        X: pd.DataFrame or np.array
            Data to explain
            
        Returns
        -------
        """
        # TODO : raise error if more than one row
        values = self.shap_values(X)
        
        return dict(zip(self.feature_names, shap_values))
    
    def explain_global(self, X):
        """
        """
        # For a tree explainer and if no example data has been provided, a lot of rows can take a while
        if (len(X) > 100) and (self.model_type == 'tree'):
            warnings.warn(f"With {len(X)} rows, this function can take a while for a tree explainer.", Warning)
        
        values = self.shap_values(X)
        values = pd.DataFrame(data=values, columns=self.feature_names).abs().mean().to_dict()
        
        return values
    
    def plot_local_explain(self, X, top=None):
        """
        """
        values = self.explain_local(X)
        plot_feature_explain(feat_importance=values, top=top)
    
    def plot_global_explain(self, X, top=None):
        """
        """
        values = self.explain_global(X)
        plot_feature_explain(feat_importance=values, top=top)
        
def plot_feature_explain(feat_importance, top=None):
    """
    """
    if type(feat_importance) == type(dict()):
        feat_importance = pd.Series(feat_importance)
    
    feat_importance = feat_importance.sort_values()
    
    if top is None:
        top = len(feat_importance)
    elif top > len(feat_importance):
        top = len(feat_importance)
    
    top_index = feat_importance.abs().sort_values().index[0:top]
    feat_importance = feat_importance.loc[top_index].sort_values()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(feat_importance.index, feat_importance.values, label=feat_importance.index, align='center')
    plt.title('Feature importance (using Shap)')
    plt.show()
