import shap
import warnings
import pandas as pd
import numpy as np

import transparentai.explainer.explainer_plots as plots


class ModelExplainer():
    """
    Class that allows to understand a local prediction or model behavior using shap_ package.
    For the moment this class will work only for model that TreeExplainer or LinearExplainer of
    shap_ package can handle.

    .. _shap : https://github.com/slundberg/shap/
    """

    global_explain = None
    X = None

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
            list of feature names (length == length of X columns)
        model_type: str (default, 'tree')
            Type of model to inspect, it can only be 'tree' or 'linear'

        Raises
        ------
        """
        if model_type not in ['tree', 'linear']:
            raise ValueError(
                'model_type has to be one of the following : tree or linear')

        self.model_type = model_type
        self.model = model

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
        values = self.explainer.shap_values(X)

        if self.model_type == 'tree':
            values = values[1]

        return values

    def explain_local(self, X, feature_classes=None):
        """
        Explain a local prediction : only one row required.

        Parameters
        ----------
        X: pd.Series or np.array
            Data to explain
        feature_classes: dict
            This dictionnary provides new values for categorical feature so
            that the feature can be more interpretable.
            dictionnary with features names as keys and for value
            a dictionnary with key, value pair representing current value
            and value to display.

        Returns
        -------
        dict:
            dictionnary with feature names as keys and
            feature importance as values

        Raises
        ------
        """
        if type(X) not in [pd.Series, np.array]:
            raise TypeError('X has to be a series or a numpy array')

        feature_names = list()
        if feature_classes is not None:
            valid_features = [
                feat for feat in feature_classes if feat in X.index]

            for feat in self.feature_names:
                if feat not in valid_features:
                    feature_names.append(f'{feat}={X.loc[feat]}')
                else:
                    feature_names.append(
                        f'{feat}={feature_classes[feat][X.loc[feat]]}')
        else:
            for feat in self.feature_names:
                feature_names.append(f'{feat}={X.loc[feat]}')

        values = self.shap_values(X)

        return dict(zip(feature_names, values))

    def explain_global(self, X):
        """
        Global explaination for a model based on a sample X
        If there are a lot of data this function could last a while.

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Data to explain    
            
        Returns
        -------
        dict:
            dictionnary with feature names as keys and
            feature importance as values 
        """
        if (self.X is not None) and type(X) in [pd.DataFrame, np.array]:
            if self.X.shape == X.shape:
                if np.all(self.X == X):
                    return 
        self.X = X

        # For a tree explainer and if no example data has been provided, a lot of rows can take a while
        if (len(X) > 100) and (self.model_type == 'tree'):
            warnings.warn(
                f"With {len(X)} rows, this function can take a while for a tree explainer.", Warning)

        if (self.feature_names is None) and (type(X) == type(pd.DataFrame())):
            self.feature_names = X.columns

        values = self.shap_values(X)
        values = pd.DataFrame(
            data=values, columns=self.feature_names).abs().mean().to_dict()

        self.global_explain = values
        return values

    def plot_local_explain(self, X, feature_classes=None):
        """
        Display a plot for a local prediction based on X set.

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Data to explain
        """
        values = self.explain_local(X, feature_classes)
        if self.model_type == 'tree':
            based_value = self.explainer.expected_value[1]
        else:
            based_value = self.explainer.expected_value

        values = self.format_feature_importance(values)

        if getattr(self.model, "predict_proba", None) is not None:
            pred = self.model.predict_proba([X])[0][1]
        else:
            pred = self.model.predict([X])[0]

        plots.plot_local_feature_influence(
            feat_importance=values, based_value=based_value, pred=pred)

    def plot_global_explain(self, X=None, top=None):
        """
        Display a plot for model global explanation based on 
        a sample X.

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Data to explain
        top: int
            top n feature to display (in case there are too
            much features)
        """
        if X is not None:
            self.explain_global(X)
        elif self.global_explain is None:
            raise ValueError(
                'Please set a X value first, using X parameter in this function or inside explain_global function')

        values = self.global_explain
        values = self.format_feature_importance(values, top=top)
        plots.plot_global_feature_influence(feat_importance=values)

    def format_feature_importance(self, feat_importance, top=None):
        """
        Format feature importance with a top value so that it returns only
        the features that have the biggest influence

        Parameters
        ----------
        feat_importance: pd.Series or dict
            current feature importance
        top: int
            number of value to get

        Returns
        -------
        pd.Series
            Feature importance formated
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

        return feat_importance
