import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score


def examine_interpretation(model, X, y, importances, epsilon=5, resolution=10, count_per_step=5, proportionality_mode=0,
                           from_zero=False, categorical_cols=[], probability_multiplier=1):
    """
    """
    baseline_predictions = model.predict(X)
    baseline_accuracy = accuracy_score(y, baseline_predictions)

    abs_importances = list(map(abs, importances))
    total_importance = (np.sum(abs_importances))
    importance_shares = list(map(lambda x: x / total_importance, abs_importances))

    if proportionality_mode == 0:
        max_importance = max(abs_importances)
        reversed_importances = list(map(lambda x: max_importance - x, abs_importances))
        total_reversed_importance = (np.sum(reversed_importances))
        inverse_importance_shares = list(map(lambda x: x / total_reversed_importance, reversed_importances))

    elif proportionality_mode == 1:
        indexes = [i for i in range(len(importances))]
        importances_df = pd.DataFrame([*zip(*[indexes, importance_shares])])
        importances_df.sort_values(by=1, axis=0, inplace=True)
        flipped_importances = pd.Series(np.flip(importances_df[1].values))
        importances_df.reset_index(inplace=True)
        importances_df[1] = flipped_importances
        importances_df.sort_values(by=0, axis=0, inplace=True)
        inverse_importance_shares = importances_df[1]

    if from_zero:
        intermediate_importances = create_intermediate_points(np.zeros(len(inverse_importance_shares)),
                                                              inverse_importance_shares, resolution)
    else:
        intermediate_importances = create_intermediate_points(inverse_importance_shares, importance_shares, resolution)

    accuraties = []

    for importances in intermediate_importances:

        this_step_accuraties = []
        for i in range(count_per_step):
            perturbed_dataset = X.copy()
            for col_idx, column in enumerate(perturbed_dataset):
                if col_idx not in categorical_cols:
                    perturbed_dataset[column] = list(
                        map(lambda x: x + importances[col_idx] * np.random.normal(0, epsilon),
                            perturbed_dataset[column]))
                else:
                    for row in perturbed_dataset[column].iteritems():
                        if np.random.random() < probability_multiplier * importances[col_idx]:
                            perturbed_dataset.at[row[0], column] = 1 - row[1]
                    perturbed_dataset[column] = np.random.permutation(perturbed_dataset[column])
            predictions = model.predict(perturbed_dataset)
            this_step_accuraties.append(accuracy_score(y, predictions))
        accuraties.append(baseline_accuracy - np.mean(this_step_accuraties))

    plt.plot(np.linspace(0, 100, resolution), accuraties)
    plt.xlabel('Percentile of perturbation range', fontsize=13)
    plt.ylabel('Loss of accuracy', fontsize=13)
    return accuraties


def examine_local_fidelity(model, X, y, epsilon=5, resolution=10, count_per_step=5, framework='shap', from_zero=False,
                           proportionality_mode=0, categorical_cols=[], probability_multiplier=1):
    """
    categorical_cols - Best result for categorical data is reached by one hot encoding and providing list of lists of column indexes of categories.
    """

    baseline_predictions = model.predict(X)
    baseline_accuracy = accuracy_score(y, baseline_predictions)
    available_frameworks = ['shap', 'lime']

    accuraties = []

    if framework == 'shap':
        explainer = shap.TreeExplainer(model)
        all_importances = explainer.shap_values(X)

        # If is multiclass, choose explanation for the correct class
        if isinstance(all_importances, list):
            right_imps = []
            for idx, label in enumerate(y):
                right_imps.append(all_importances[label][idx])
            all_importances = right_imps
    #         explainer = shap.KernelExplainer(model, data=X)
    #         all_importances = explainer.shap_values(X)

    elif framework == 'lime':
        all_importances = []
        explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns)

        for index, row in X.iterrows():
            correct_label = y[index]
            if np.shape(correct_label):
                correct_label = correct_label.iloc[0]
            # If is multiclass, choose explanation for the correct class
            exp = explainer.explain_instance(row, model.predict_proba, num_features=len(X.columns),
                                             labels=(correct_label,))
            imps = dict()

            for feat in exp.local_exp[correct_label]:
                imps[feat[0]] = feat[1]
            imp_vals = []
            for i in range(len(imps)):
                imp_vals.append(imps[i])

            all_importances.append(imp_vals)

    else:
        raise ValueError("Framework not found")

    abs_imps = [list(map(abs, row)) for row in all_importances]
    imp_shares = [list(map(lambda x: x / sum(row), row)) for row in abs_imps]

    if proportionality_mode == 0:
        reversed_importances = [list(map(lambda x: max(row) - x, row)) for row in abs_imps]
        reversed_imp_shares = [list(map(lambda x: x / sum(row), row)) for row in reversed_importances]
    elif proportionality_mode == 1:
        reversed_imp_shares = []
        indexes = [i for i in range(len(imp_shares[0]))]
        for single_obs_imp in imp_shares:
            importances_df = pd.DataFrame([*zip(*[indexes, single_obs_imp])])
            importances_df.sort_values(by=1, axis=0, inplace=True)
            flipped_importances = pd.Series(np.flip(importances_df[1].values))
            importances_df.reset_index(inplace=True)
            importances_df[1] = flipped_importances
            importances_df.sort_values(by=0, axis=0, inplace=True)
            reversed_imp_shares.append(importances_df[1])

    if from_zero:
        imps_matrix = np.linspace(np.zeros(np.shape(reversed_imp_shares)), reversed_imp_shares, resolution)
    else:
        imps_matrix = np.linspace(reversed_imp_shares, imp_shares, resolution)

    accuraties = []

    # One hot encoded category handling to be added
    # all_categorical_cols = list(chain(*categorical_cols))

    for step_importances in imps_matrix:
        this_step_accuraties = []
        for i in range(count_per_step):
            perturbed_dataset = X.copy()
            for index, (skip, row) in enumerate(perturbed_dataset.iterrows()):
                for idx in range(len(row)):
                    # perturb numerical data
                    if idx not in categorical_cols:
                        row[idx] = row[idx] + step_importances[index][idx] * np.random.normal(0, epsilon)
                    # change randomly categorical data
                    else:
                        if np.random.random() < probability_multiplier * step_importances[index][idx]:
                            row[idx] = 1 - row[idx]

            predictions = model.predict(perturbed_dataset)
            this_step_accuraties.append(accuracy_score(y, predictions))
        accuraties.append(baseline_accuracy - np.mean(this_step_accuraties))

    plt.plot(np.linspace(0, 100, resolution), accuraties)
    plt.xlabel('Percentile of perturbation range', fontsize=13)
    plt.ylabel('Loss of accuracy', fontsize=13)
    return accuraties


def create_intermediate_points(start_vals, end_vals, resolution):
    arr = []
    for start_val, end_val in zip(start_vals, end_vals):
        arr.append(np.linspace(start_val, end_val, resolution))
    return [*zip(*arr)]


def get_lipschitz(model, X, y, epsilon=3, framework='shap', sample_num=None):
    if not sample_num:
        sample_num = int(len(X) / 2)

    indexes = np.random.permutation(len(X))[:sample_num]

    data = X.iloc[indexes]

    if framework == 'shap':
        explainer = shap.TreeExplainer(model)
        all_importances = explainer.shap_values(X)

        # If is multiclass, choose explanation for the correct class
        if isinstance(all_importances, list):
            right_imps = []
            for idx, label in enumerate(y):
                right_imps.append(all_importances[label][idx])
            all_importances = right_imps
    #         explainer = shap.KernelExplainer(model, data=X)
    #         all_importances = explainer.shap_values(X)

    elif framework == 'lime':
        all_importances = []

        explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns)

        for index, (skip, row) in enumerate(X.iterrows()):
            correct_label = y[index]

            # If is multiclass, choose explanation for the correct class
            exp = explainer.explain_instance(row, model.predict_proba, num_features=len(X.columns),
                                             labels=(correct_label,))
            imps = dict()

            for feat in exp.local_exp[correct_label]:
                imps[feat[0]] = feat[1]
            imp_vals = []
            for i in range(len(imps)):
                imp_vals.append(imps[i])

            all_importances.append(imp_vals)

    else:
        print('Bad framework.')
        return None

    if isinstance(indexes, np.ndarray):
        indexes = indexes.tolist()

    #     print(all_importances)
    #     print(type(all_importances))
    #     print(indexes)
    #     print(type(indexes))

    #    chosen_imps = all_importances[indexes]
    l_values = []

    for data_idx, (skip, observation) in enumerate(data.iterrows()):
        max_val = 0
        for idx, (skip, other_observation) in enumerate(X.iterrows()):
            dist = np.linalg.norm(observation - other_observation)
            if dist < epsilon:
                l_val = np.linalg.norm(
                    pd.core.series.Series(all_importances[indexes[data_idx]]) - pd.core.series.Series(
                        all_importances[idx])) / dist

                if l_val > max_val:
                    max_val = l_val
        if max_val:
            l_values.append(max_val)
    return l_values


def check_consistency(models, X, y, epsilon=3, framework='shap', sample_num=None):
    if not isinstance(models, list) or len(models) < 2:
        print('Provide list of models as the first argument')
        return

    """
    sample_num - how many observations should be compared across the models 
    """

    if not sample_num:
        sample_num = int(len(X) / 4)

    indexes = np.random.permutation(len(X))[:sample_num]

    data = X.iloc[indexes]

    chosen_importances_per_model = []

    if framework == 'shap':
        for model in models:
            explainer = shap.TreeExplainer(model)
            all_importances = explainer.shap_values(X)

            # If is multiclass, choose explanation for the correct class
            if isinstance(all_importances, list):
                right_imps = []
                for idx, label in enumerate(y):
                    right_imps.append(all_importances[label][idx])
                all_importances = right_imps

            chosen_importances = [all_importances[i] for i in indexes]
            chosen_importances_per_model.append(chosen_importances)


    elif framework == 'lime':
        for model in models:
            all_importances = []

            explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns)

            for index, (skip, row) in enumerate(data.iterrows()):
                correct_label = y[index]

                # If is multiclass, choose explanation for the correct class
                exp = explainer.explain_instance(row, model.predict_proba, num_features=len(X.columns),
                                                 labels=(correct_label,))
                imps = dict()

                for feat in exp.local_exp[correct_label]:
                    imps[feat[0]] = feat[1]
                imp_vals = []
                for i in range(len(imps)):
                    imp_vals.append(imps[i])

                all_importances.append(imp_vals)
            chosen_importances_per_model.append(all_importances)

    else:
        print('Bad framework.')
        return None

    if isinstance(indexes, np.ndarray):
        indexes = indexes.tolist()

    c_values = []

    for obs_idx in range(len(data)):
        largest_dist = 0
        for model_idx, model_imps in enumerate(chosen_importances_per_model):
            for compared_model in chosen_importances_per_model[:model_idx]:
                current_imps = model_imps[obs_idx]
                other_imps = compared_model[obs_idx]
                if not isinstance(current_imps, np.ndarray):
                    current_imps = np.array(current_imps)
                if not isinstance(other_imps, np.ndarray):
                    other_imps = np.array(other_imps)
                dist = np.linalg.norm(current_imps - other_imps)
                if dist > largest_dist:
                    largest_dist = dist
        c_values.append(largest_dist)

    return c_values


def gradual_elimination(model, X, y, importances, test_X=None, test_y=None, single_drop=False, categorical_cols=[],
                        removal_order=[]):
    # Sort importances while keeping corresponding feature information
    indexes = [i for i in range(len(importances))]
    importances_df = pd.DataFrame([*zip(*[indexes, importances])])
    importances_df['Feature names'] = X.columns

    if not removal_order:
        importances_df.sort_values(by=1, axis=0, inplace=True, ascending=True)
    else:
        importances_df['Removal order'] = removal_order
        importances_df.sort_values(by='Removal order', axis=0, inplace=True, ascending=True)

    if test_X:
        limited_X = test_X.copy()
    else:
        test_X = X.copy()
        test_y = y.copy()
        limited_X = test_X.copy()

    baseline_accuracy = accuracy_score(test_y, model.predict(test_X))
    accuracy_losses = []

    for feat_idx in importances_df[0]:

        # Bigger data set is used to get expected values.
        original_col = X[X.columns[feat_idx]]

        # Categorical data is accepted as 0's and 1's column, multiclass yet to be added. Proportion is
        if feat_idx in categorical_cols:
            limited_X[limited_X.columns[feat_idx]] = np.random.permutation(limited_X[limited_X.columns[feat_idx]])
        else:
            limited_X[limited_X.columns[feat_idx]] = np.mean(X[X.columns[feat_idx]])

        acc_loss = baseline_accuracy - accuracy_score(test_y, model.predict(limited_X))
        accuracy_losses.append(acc_loss)

        if single_drop:
            limited_X = test_X.copy()

    sns.set_style("whitegrid")

    fig = plt.figure()

    importances_df.reset_index(drop=True, inplace=True)
    importances_df['Loss of accuracy'] = accuracy_losses

    ax1 = fig.add_subplot(111)
    sns.barplot(x=importances_df['Feature names'], y=importances_df[1], edgecolor='k', color='aliceblue', linewidth=2,
                order=importances_df[0])
    plt.ylabel('Importance share')
    plt.xlabel('Dropped feature name')

    sns.set_style("white")
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    sns.lineplot(x=[i for i in range(len(importances))], y=importances_df['Loss of accuracy'], marker='o', color='m',
                 linewidth=2)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel("Loss of accuracy", color='m')
    plt.ylim((0, 0.6))

    return importances_df
