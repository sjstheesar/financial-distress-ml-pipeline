import itertools
import warnings
import pickle
import datetime
import csv
import numpy as np

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              BaggingClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from viz import (plot_predicted_scores, plot_precision_recall, plot_auc_roc,
                 plot_feature_importances)

warnings.filterwarnings("ignore")


INPUT_DIR = "../processed_data/"
OUTPUT_DIR = "../log/"
OUTPUT_FILE = "performances.csv"

MODEL_NAMES = ["KNN", "Logistic Regression", "Decision Tree", "Linear SVM",
               "Bagging", "Boosting", "Random Forest"]
MODELS = [KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier,
          LinearSVC, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier]

METRICS_NAMES = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC ROC Score"]
METRICS = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]

THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SEED = 123

GRID_SEARCH_PARAMS = {"KNN": {
                              'n_neighbors': list(range(50, 110, 20)),
                              'weights': ["uniform", "distance"],
                              'metric': ["euclidean", "manhattan", "minkowski"]
                              },

                      "Logistic Regression": {
                                              'penalty': ['l1', 'l2'],
                                              'C': [0.001, 0.01, 0.1, 1, 10],
                                              'solver': ['newton-cg', 'lbfgs', 'liblinear']
                                              },

                      "Decision Tree": {
                            'criterion': ["entropy", "gini"],
                            'min_samples_split': list(np.arange(0.02, 0.05, 0.01)),
                            'max_depth': list(range(4, 11)),
                            'max_features': list(range(4, 15, 2))
                            },

                      "Linear SVM": {
                                     'penalty': ['l1', 'l2'],
                                     'C': [0.001, 0.01, 0.1, 1, 10]
                                     },

                      "Bagging": {
                                  'max_samples': [0.05, 0.1, 0.2, 0.5],
                                  'max_features': list(range(4, 15, 2))
                                  },

                      "Boosting":{
                                  'algorithm': {"SAMME", "SAMME.R"},
                                  'learning_rate': [0.001, 0.01, 0.1, 0.5, 1, 10]
                                  },

                      "Random Forest": {
                            'min_samples_split': list(np.arange(0.01, 0.06, 0.01)),
                            'max_depth': list(range(4, 11)),
                            'max_features': list(range(4, 15, 2))
                            }
                      }

DEFAULT_ARGS = {"KNN": {'n_jobs': -1},
                "Logistic Regression": {'random_state': SEED},
                "Decision Tree": {'random_state': SEED},
                "Linear SVM": {'random_state': SEED, 'max_iter': 200},
                "Bagging": {'n_estimators': 50, 'random_state': SEED,
                            'oob_score': True, 'n_jobs': -1},
                "Boosting": {'n_estimators': 100, 'random_state': SEED},
                "Random Forest": {'n_estimators': 300, 'random_state': SEED,
                                  'oob_score': True, 'n_jobs': -1}}


SNAP_SHOTS_COUNT = 0


#----------------------------------------------------------------------------#
def ask_user():
    """
    Ask the user for a classifier and metric index.
    """
    print("Up till now we support:\n")
    for i in range(len(MODEL_NAMES)):
        print("{}. {}".format(i, MODEL_NAMES[i]))
    model_index = int(input(("We use default Decision Tree as the benchmark.\n"
                             "Please input a classifier index:\n")))

    print(("Up till now we use the following metrics to evaluate the"
           " fitted classifiers on the validation and test set.\n"))
    for i in range(len(METRICS)):
        print("{}. {}".format(i, METRICS_NAMES[i].title()))
    metric_index = int(input("Please input a metrics index:\n"))

    return model_index, metric_index


def load_preprocessed_features(dir_path=INPUT_DIR, test=True):
    """
    Load pre-processed feature matrices.
    """
    try:
        Xs = np.load(dir_path + 'X.npz')
        ys = np.load(dir_path + 'y.npz')

        if not test:
            X_train = Xs['train']
            y_train = ys['train']
            return X_train, y_train

        X_train, X_test = Xs['train'], Xs['test']
        y_train, y_test = ys['train'], ys['test']
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading preprocessed features: {e}")
        return None


def build_default_benchmark(data, metric_index):
    """
    Build and evaluate the default decision tree benchmark model.
    """
    try:
        X_train, X_test, y_train, y_test = data

        benchmark = DecisionTreeClassifier(**DEFAULT_ARGS["Decision Tree"])
        benchmark.fit(X_train, y_train)
        predicted_probs = benchmark.predict_proba(X_test)[:, 1]
        benchmark_score = METRICS[metric_index](y_test, benchmark.predict(X_test))

        print("\n{} of the default decision tree model is {:.4f}.\n".\
              format(METRICS_NAMES[metric_index], round(benchmark_score, 4)))

        return benchmark_score
    except Exception as e:
        print(f"Error building default benchmark: {e}")
        return None


def predict_probabilities(clf, X_test):
    """
    Predict probabilities for the test set.
    """
    try:
        if hasattr(clf, "predict_proba"):
            predicted_prob = clf.predict_proba(X_test)[:, 1]
        else:
            prob = clf.decision_function(X_test)
            predicted_prob = (prob - prob.min()) / (prob.max() - prob.min())

        return predicted_prob
    except Exception as e:
        print(f"Error predicting probabilities: {e}")
        return None


def perform_cross_validation(clf, skf, data, metric_index, threshold):
    """
    Perform cross-validation and evaluate the model.
    """
    try:
        X_train, y_train = data
        predicted_probs, scores = [], []

        for train, validation in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train], X_train[validation]
            y_tr, y_val = y_train[train], y_train[validation]

            clf.fit(X_tr, y_tr)
            predicted_prob = predict_probabilities(clf, X_val)
            predicted_labels = np.where(predicted_prob > threshold, 1, 0)

            predicted_probs.append(predicted_prob)
            scores.append(METRICS[metric_index](y_val, predicted_labels))

        return list(itertools.chain(*predicted_probs)), np.array(scores).mean()
    except Exception as e:
        print(f"Error performing cross-validation: {e}")
        return None, 0


def find_best_threshold(model_index, metric_index, train_data,
                        verbose=True, plot=True):
    """
    Find the best threshold for the model.
    """
    try:
        model_name = MODEL_NAMES[model_index]
        metric_name = METRICS_NAMES[metric_index]
        default_args = DEFAULT_ARGS[model_name]

        clf = MODELS[model_index](**default_args)
        skf = StratifiedKFold(n_splits=5, random_state=SEED)

        if plot:
            default_probs, _ = perform_cross_validation(clf, skf, train_data, metric_index, 0.5)
            plot_predicted_scores(default_probs, "({} -- {})".format(model_name, metric_name))

        best_score, best_threshold = 0, None
        print("Default {}. Search Starts:".format(model_name))
        for threshold in THRESHOLDS:
            _, score = perform_cross_validation(clf, skf, train_data, metric_index, threshold)
            if verbose:
                print("\t(Threshold: {}) the cross-validation {} is {:.4f}".\
                      format(threshold, metric_name, score))
            if score > best_score:
                best_score, best_threshold = score, threshold

        print("Search Finished: The best threshold to use is {:.4f}.\n".format(best_threshold))
        return best_threshold
    except Exception as e:
        print(f"Error finding best threshold: {e}")
        return None


def tune_hyperparameters(model_index, metric_index, train_data, best_threshold,
                         n_folds=10, verbose=True):
    """
    Tune hyperparameters using grid search and cross-validation.
    """
    try:
        model_name = MODEL_NAMES[model_index]
        metric_name = METRICS_NAMES[metric_index]
        params_grid = GRID_SEARCH_PARAMS[model_name]
        default_args = DEFAULT_ARGS[model_name]

        best_score, best_grid = 0, None
        params = params_grid.keys()
        skf = StratifiedKFold(n_splits=n_folds, random_state=SEED)

        print("{} with Decision Threshold {}. Search Starts:".format(model_name,
                                                                     best_threshold))
        for grid in itertools.product(*(params_grid[param] for param in params)):
            args = dict(zip(params, grid))
            clf = MODELS[model_index](**default_args, **args)
            _, grid_score = perform_cross_validation(clf, skf, train_data, metric_index, best_threshold)

            if grid_score > best_score:
                best_score, best_grid = grid_score, args
            if verbose:
                print("\t(Parameters: {}), cross-validation {} of {:.4f}".format(args, metric_name,
                                                                                 grid_score))

        print("Search Finished: The best parameters to use is {}\n".format(best_grid))
        return best_grid, best_score
    except Exception as e:
        print(f"Error tuning hyperparameters: {e}")
        return None, 0


def calculate_precision_at_k(y_true, y_scores, k):
    """
    Calculate precision at a given threshold.
    """
    try:
        idx = np.argsort(np.array(y_scores))[::-1]
        y_scores, y_true = np.array(y_scores)[idx], np.array(y_true)[idx]
        cutoff_index = int(len(y_scores) * (k / 100.0))
        preds_at_k = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]

        return precision_score(y_true, preds_at_k)
    except Exception as e:
        print(f"Error calculating precision at k: {e}")
        return None


def evaluate_tuned_model(model_index, metric_index, best_threshold, best_grid, data,
                         plot=True, verbose=True, write=False, output=None):
    """
    Evaluate the tuned model on the test set.
    """
    try:
        X_train, X_test, y_train, y_test = data
        model_name = MODEL_NAMES[model_index]
        metric_name = METRICS_NAMES[metric_index]
        default_args = DEFAULT_ARGS[model_name]

        clf = MODELS[model_index](**default_args, **best_grid)
        clf.fit(X_train, y_train)

        predicted_prob = predict_probabilities(clf, X_test)
        predicted_labels = np.where(predicted_prob > best_threshold, 1, 0)
        test_score = METRICS[metric_index](y_test, predicted_labels)
        print(("Our {} classifier reached a(n) {} of {:.4f} with a decision"
               " threshold of {} on the test set.\n").format(model_name, metric_name,
                                                             test_score, best_threshold))
        if write:
            log = [model_name, best_threshold, best_grid, default_args]
            log += [metric(y_test, predicted_labels) for metric in METRICS]

            pred_prob_sorted, y_testsorted = zip(*sorted(zip(predicted_prob, y_test),
                                                     reverse=True))
            log += [calculate_precision_at_k(y_testsorted, pred_prob_sorted, threshold)
                    for threshold in THRESHOLDS]

            output.append(log)

        if plot:
            positives = np.count_nonzero(np.append(y_train, y_test))
            baseline = positives / (len(y_test) + len(y_train))
            plot_precision_recall(y_test, predicted_prob, baseline, "({} -- {})".\
                                  format(model_name, metric_name))

            plot_auc_roc(clf, data, "({} -- {})".format(model_name, metric_name))

            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                with open(INPUT_DIR + 'col_names.pickle', 'rb') as handle:
                    col_names = pickle.load(handle)
                plot_feature_importances(importances, col_names, title="({} -- {})".\
                                         format(model_name, metric_name))

        return test_score
    except Exception as e:
        print(f"Error evaluating tuned model: {e}")
        return None


def train_and_evaluate_model(model_index, metric_index, data, train_data,
                             write=False, output=None):
    """
    Train and evaluate a specific model.
    """
    try:
        metric_name = METRICS_NAMES[metric_index]
        model_name = MODEL_NAMES[model_index]

        benchmark_score = build_default_benchmark(data, metric_index)
        best_threshold = find_best_threshold(model_index, metric_index, train_data)
        best_grid, _ = tune_hyperparameters(model_index, metric_index, train_data, best_threshold)

        if write:
            test_score = evaluate_tuned_model(model_index, metric_index,
                                             best_threshold, best_grid, data,
                                             write=True, output=output)
        else:
            test_score = evaluate_tuned_model(model_index, metric_index,
                                             best_threshold, best_grid, data)

        diff = round(test_score - benchmark_score, 4)
        print(("{} of the tuned {} is {}, {} {} than the benchmark.\n"
               "**-------------------------------------------------------------**\n\n").\
           format(metric_name, model_name, round(test_score.mean(), 4), diff,
                  ['higher', 'lower'][int(diff <= 0)]))
    except Exception as e:
        print(f"Error training and evaluating model: {e}")


#----------------------------------------------------------------------------#
if __name__ == "__main__":
    try:
        print(("You can either choose to train with a specific configuration (input 1),"
               " or all the models by metrics we have (input 2)"))
        flag = int(input("Please input your choice:\n"))

        X_train, X_test, y_train, y_test = load_preprocessed_features()
        data = [X_train, X_test, y_train, y_test]
        train_data = [X_train, y_train]

        if flag == 1:
            model_index, metric_index = ask_user()
            train_and_evaluate_model(model_index, metric_index, data, train_data)
        else:
            header = ["Model", "Threshold", "Hyperparameters", "Default Parameters"]
            header += METRICS_NAMES
            header += ["p_at_{}".format(threshold) for threshold in THRESHOLDS]
            logs = [header]

            for model_index in range(len(MODELS)):
                for metric_index in range(len(METRICS)):
                    print(("**-------------------------------------------------------------**\n"
                           "Training for {} with metric {}.").format(MODEL_NAMES[model_index],
                                                                     METRICS_NAMES[metric_index]))
                    train_and_evaluate_model(model_index, metric_index, data, train_data,
                                               write=True, output=logs)

                    with open(OUTPUT_DIR + 'snapshots' + '{}.pickle'.format(SNAP_SHOTS_COUNT), 'wb') as handle:
                        pickle.dump(logs, handle)
                    SNAP_SHOTS_COUNT += 1
        
            with open(OUTPUT_DIR + OUTPUT_FILE, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(set(logs))

        _ = input("Press any key to exit.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
