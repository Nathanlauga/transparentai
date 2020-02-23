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


    Example
    -------

    For binary classification :

    X attribute is optional but if you have a lot of data it improves 
    compute times using 100 rows sample.

    >>> from transparentai.explainer import ModelExplainer
    >>> explainer = ModelExplainer(model=clf, X=X_train, model_type='tree')

    For multi labels classification : 

    >>> explainer = ModelExplainer(model=clf, X=X_train, model_type='tree',
                                   multi_label=True)

    For regression : 

    Here X attribute is mandatory.

    >>> explainer = ModelExplainer(model=reg, X=X_train, model_type='linear')

    Computes feature importance with shap.

    >>> explainer.explain_global(X_test)
    {'CRIM': 0.506476821156253,
    'ZN': 0.7480023859571747,
    'INDUS': 0.12914193686288905,
    'CHAS': 0.34789486135074216,
    'NOX': 1.6113838160264444,
    'RM': 2.0451339865520524,
    'AGE': 0.015167695255383062,
    'DIS': 2.447244637853398,
    'RAD': 2.3718108812309993,
    'TAX': 1.8757421732944977,
    'PTRATIO': 1.7010831097216834,
    'B': 0.5731631851895846,
    'LSTAT': 2.6035683651209123}

    For more details please see the `ModelExplainer for binary classification`_, 
    `ModelExplainer for multi labels classification`_ or `ModelExplainer for regression`_ notebooks.

    .. _ModelExplainer for binary classification : https://github.com/Nathanlauga/transparentai/notebooks/example_ModelExplainer_binary_classification.ipynb
    .. _ModelExplainer for multi labels classification : https://github.com/Nathanlauga/transparentai/notebooks/example_ModelExplainer_multi_label_classification.ipynb
    .. _ModelExplainer for regression : https://github.com/Nathanlauga/transparentai/notebooks/example_ModelExplainer_regression.ipynb

    Attributes
    ----------
    model_type:
        Type of model to inspect, it can only be 'tree' or 'linear'
    model:
        model to inspect
    multi_label: bool
        Whether there is more than 2 classes in the label column
        (only for classification)
    explainer: shap.TreeExplainer or shap.LinearExplainer
        explainer object that has expected values and can
        compute shap values 
    feature_names: np.array
        list of feature names (length == length of X columns)
    global_explain: dict
        dictionnary with feature names as keys and
        global feature importance as values 
    X: pd.DataFrame or pd.Series
        Data to explain with column names as indexes / columns
    """

    global_explain = None
    X = None
    feature_names = None

    def __init__(self, model, X=None, model_type='tree', multi_label=False):
        """
        Parameters
        ----------
        model:
            model to inspect
        X: pd.DataFrame (default, None)
            data (possibly training) to start the explainer
            mandatory for a linear model but for tree model it can improve computing time
            so it's recommanded to use it.
        model_type: str (default 'tree')
            Type of model to inspect, it can only be 'tree' or 'linear'
        multi_label: bool
            Whether there is more than 2 classes in the label column
            (only for classification)

        Raises
        ------
        ValueError:
            model_type has to be one of the following : 'tree' or 'linear'
        TypeError:
            X has to be a pandas.DataFrame
        ValueError:
            If you use a linear explainer then X should be set.
        """
        if model_type not in ['tree', 'linear']:
            raise ValueError(
                'model_type has to be one of the following : tree or linear')
        if type(X) is not pd.DataFrame:
            raise TypeError('X has to be a pandas.DataFrame')

        self.model_type = model_type
        self.model = model
        self.multi_label = multi_label

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

        if X is not None:
            self.feature_names = X.columns.values

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

        if (self.model_type == 'tree') & (not self.multi_label):
            values = values[1]

        return values

    def explain_local(self, X, feature_classes=None):
        """
        Explain a local prediction : only one row required.

        Parameters
        ----------
        X: pd.Series
            Data to explain with column names as indexes
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
        if type(X) not in [pd.Series]:
            raise TypeError('X has to be a series or a numpy array')

        if self.feature_names is None:
            self.feature_names = X.index.values

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

        if not self.multi_label:
            return dict(zip(feature_names, values))
        else:
            shap_dict = dict()
            for feat in feature_names:
                shap_dict[feat] = list()

            for i in range(0, len(values)):
                for j, feat in enumerate(feature_names):
                    shap_dict[feat].append(values[i][j])
            return shap_dict

    def explain_global(self, X):
        """
        Global explaination for a model based on a sample X
        If there are a lot of data this function could last a while.

        Parameters
        ----------
        X: pd.DataFrame or pd.Series
            Data to explain    

        Returns
        -------
        dict:
            dictionnary with feature names as keys and
            feature importance as values 
        """
        if (self.X is not None) and type(X) in [pd.DataFrame, pd.Series]:
            if self.X.shape == X.shape:
                if np.all(self.X.values == X.values):
                    return self.global_explain
        self.X = X

        # For a tree explainer and if no example data has been provided, a lot of rows can take a while
        if (len(X) > 100) and (self.model_type == 'tree'):
            warnings.warn(
                f"With {len(X)} rows, this function can take a while for a tree explainer.")

        if (self.feature_names is None) and (type(X) == type(pd.DataFrame())):
            self.feature_names = X.columns.values
        elif type(X) == pd.Series:
            self.feature_names = X.index.values

        values = self.shap_values(X)

        if not self.multi_label:
            values = pd.DataFrame(
                data=values, columns=self.feature_names).abs().mean().to_dict()
        else:
            tmp = dict()
            for i in range(0, len(values)):
                tmp[i] = pd.DataFrame(
                    data=values[i], columns=self.feature_names).abs().mean().to_dict()
            values = tmp
            del tmp

        self.global_explain = values
        return values

    def _get_base_value(self):
        """
        Returns base value from the explainer attribute.

        Returns
        -------
        np.array
            Array with all expected values. If it's a tree explainer and
            a binary classification then it returns for only the class 1
        """
        if (self.model_type == 'tree') & (not self.multi_label):
            return self.explainer.expected_value[1]
        else:
            return self.explainer.expected_value

    def _get_predictions(self, X, num_class=1):
        """
        Returns the prediction using `predict_proba` function if it exists
        else it uses predict function. 

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Data to explain
        num_class: int (default 1)
            Class number for which we want to see the explanation
            if it's a binary classification then the value is 1

        Returns
        -------
        number:
            prediction
        """
        if getattr(self.model, "predict_proba", None) is not None:
            return self.model.predict_proba([X])[0][num_class]
        else:
            return self.model.predict([X])[0]

    def _plot_local_explain(self, X, values, base_value, top=None, num_class=1):
        """
        Display local feature influence sorted for a specific
        prediction.
        
        Parameters
        ----------
        X: pd.DataFrame or np.array
            Data to explain
        values: pd.Series
            Feature importance with feature as indexes and 
            shap value as values
        base_value: number
            prediction value if we don't put any feature into the model
        top: int
            top n feature to display (in case there are too
            much features)
        num_class: int (default 1)
            Class number for which we want to see the explanation
            if it's a binary classification then the value is 1
        """
        values = self.format_feature_importance(values, top=top)
        pred = self._get_predictions(X, num_class=num_class)

        plots.plot_local_feature_influence(
            feat_importance=values, base_value=base_value, pred=pred)

    def plot_local_explain(self, X, feature_classes=None, top=None):
        """
        Display a plot for a local prediction based on X set.

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Data to explain
        feature_classes: dict
            This dictionnary provides new values for categorical feature so
            that the feature can be more interpretable.
            dictionnary with features names as keys and for value
            a dictionnary with key, value pair representing current value
            and value to display.
        """
        values = self.explain_local(X, feature_classes)

        base_value = self._get_base_value()

        if not self.multi_label:
            self._plot_local_explain(X, values, base_value, top=top)
        else:
            for i, base_val in enumerate(base_value):
                val = {}
                for k, v in values.items():
                    val[k] = v[i]

                print(f'Plot for the {i}th class probability.')
                self._plot_local_explain(X, val, base_val, top=top, num_class=i)

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

        Raises
        ------
        AttributeError:
            If X parameter is None then you have to add X in explain_global function first
            or directly in this function if you prefer to plot directly.
        """
        if X is not None:
            self.explain_global(X)
        elif self.global_explain is None:
            raise AttributeError(
                'Please set a X value first, using X parameter in this function or inside explain_global function')

        values = self.global_explain
        if not self.multi_label:
            values = self.format_feature_importance(values, top=top)
            plots.plot_global_feature_influence(feat_importance=values)
        else:
            for i in range(0, len(values)):
                val = self.format_feature_importance(values[i], top=top)
                print(f'Plot for the {i}th class.')
                plots.plot_global_feature_influence(feat_importance=val)

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

        top_index = feat_importance.abs().sort_values(
            ascending=False).index[0:top]
        other_index = feat_importance.abs().sort_values(
            ascending=False).index[top:]

        feat_importance_filtered = feat_importance.loc[top_index].sort_values()
        if len(other_index) > 0:
            feat_importance_filtered['Others variables'] = feat_importance.loc[other_index].sum(
            )

        return feat_importance_filtered.sort_values()