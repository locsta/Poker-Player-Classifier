import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from IPython.display import Image  
from pydotplus import graph_from_dot_data
from xgboost import plot_tree

def plot_confusion_matrix(labels, preds_proba, classes, threshold=0.5, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Calculate the confusion matrix for a determined threshold and plot it in a nicely formatted way

    Parameters:
    labels (list): List of labels (class)
    preds_proba (list): List of predicted probabilities
    threshold (float): Threshold
    normalize (bool): normalize True will will plot using percentages, normalize False will plot using raw values
    title (str): Plot's title
    cmap (str): Color Map
    return_cm (bool): return the non-normalised confusion matrix if set to True


    Returns:
    float:Threshold optimal value

   """
    preds = preds_proba_to_preds_class(preds_proba,threshold)
    cm_std = confusion_matrix(labels, preds)
    # Check if normalize is set to True
    # If so, normalize the raw confusion matrix before visualizing
    if normalize:
        cm = cm_std.astype('float') / cm_std.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm = cm_std
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, cmap=cmap)
    
    # Add title and axis labels 
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Add appropriate axis scales
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Text formatting
    fmt = '.2f' if normalize else 'd'
    # Add labels to each cell
    thresh = cm.max() / 2.
    # Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    # Add a legend
    plt.colorbar()
    plt.show()
    return cm_std
    
def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    """Find the best value for K in a certain range of values for KNeighborsClassifier

    Parameters:
    X_train (float): The percentage of positives in the population
    y_train (float): The Cost of false positives minus the cost of true negatives
    X_test (float): The Cost of false negatives minus the cost of true positives
    y_test (float): True positive rate
    min_k (float): Minimum value for K
    max_k (float): Maximum value for K


    Returns:
    int: Best value for K

   """
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds)
        if f1 > best_score:
            best_k = k
            best_score = f1
    print("Best Value for k: {}".format(best_k))
    print("F1-Score: {}".format(best_score))

def preds_proba_to_preds_class(preds_proba,threshold):
    """Transform prediction probabilities into classes (booleans) using a determined threshold

    Parameters:
    preds_proba (list): List of probabilities
    threshold (float): Threshold

    Returns:
    classes (list): List of classes

   """
    return [True if pred > threshold else False for pred in preds_proba]

def threshold_selection(prevalence, CostFP_minus_CostTN, CostFN_minus_CostTP, y, y_hat):
    """Calculate the optimal treshold depending on prevalence, costs, true positive rates and false positive rates
    
    Args:
        prevalence (float): The percentage of positives in the population
        CostFP_minus_CostTN (float): The cost of false positives minus the cost of true negatives
        CostFN_minus_CostTP (float): The cost of false negatives minus the cost of true positives
        y (list): True labels (classes)
        y_hat (list): Predicted proba for labels (classes)
    
    Returns:
        [float]: Best threshold
    """
    fpr, tpr, thresholds = roc_curve(y, y_hat)
    m = ((1 - prevalence) / prevalence) * ((CostFP_minus_CostTN) / (CostFN_minus_CostTP))
    fm_thresholds = []
    for i in range(len(fpr)):
        fm = tpr[i] - (m * fpr[i])
        fm_thresholds.append((thresholds[i], fm))
    fm_thresholds = sorted(fm_thresholds, key=lambda fm_value: fm_value[1], reverse=True)
    return fm_thresholds[0][0]

def metrics(labels, preds_proba, print_metrics=True, plot=False, threshold=0.5, rounded=4):
    """Plot the ROC curve, calculate and print AUC, Precision, Recall, Accuracy and F1 scores

    Parameters:
    labels (list): List of labels (classes)
    preds_proba (list): List of predicted probabilities
    print_metrics (bool): Print the metrics if parameter set to True
    plot (bool): Plot the ROC curve if parameter set to True
    threshold (float): Threshold (purely as information)
    rounded (int): The number of digits for the scores


    Returns:
    scores (dict): Return a dictionnary of scores (AUC, Precision, Recall, Accuracy and F1)

   """
    

    test_fpr, test_tpr, test_thresholds = roc_curve(labels, preds_proba)
    roc_auc = auc(test_fpr, test_tpr)
    preds = preds_proba_to_preds_class(preds_proba,threshold)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    if plot:
        sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(test_fpr, test_tpr, color='darkorange',
                lw=lw, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.yticks([i/20.0 for i in range(21)])
        plt.xticks([i/20.0 for i in range(21)])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
    if print_metrics:
        print(f"ROC AUC Score: {roc_auc}\n")
        print(f"------- Metrics for threshold {threshold} -------")
        print(f"- Precision Score: {precision}")
        print(f"- Recall Score: {recall}")
        print(f"- Accuracy Score: {accuracy}")
        print(f"- F1 Score: {f1}\n")
    else:
        return {"roc_auc":roc_auc, "precision":precision, "recall":recall, "accuracy":accuracy, "f1":f1}


def get_roc_auc(y_test, y_hat_test):
    """Return the ROC AUC value

    Parameters:
    y_test (list): Target Labels
    y_hat_test (list): Predicted target labels

    Returns:
    float:Area Under The Curve pof Receiver Operating Characteristics

   """
    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_hat_test)
    roc_auc = auc(test_fpr, test_tpr)
    return roc_auc

def print_corr(df, pct=0):
    """Prints the multicollinearity heatmap with the option of setting a minimum multicollinearity percentage

    Parameters:
    df (pandas dataframe): Pandas DataFrame
    pct (float): Optional minimum multicollinearity percentage (multicollinearity lower than the value of the variable "pct" will not be shown)

    Returns:
    void:This function does not return values

   """
    
    sns.set(style="white")

    # Compute the correlation matrix
    if pct == 0:
        corr = df.corr()
    else:
        corr = abs(df.corr()) > pct

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})


def prediction(X, model, stakes="default"):
    """The function will predict the class of a player depending on his stats and on the stakes of the game we are playing
    
    Args:
        X (dataframe): Player's stat
        model (classifier object): A fitted classifier model
        stakes (str): Stakes of the game (small or high)
    """
    
    #Defining thresholds
    thresholds = {"default":0.5, "small":0.696, "high":0.326}

    # Chosing default threshold if value entered for stakes ins't a key of the threshold dictionnary
    if stakes not in thresholds.keys():
        stakes = "default"
        print(f"The value entered for stakes isn't recognized, therefore threshold default value: {thresholds[stakes]} was chosen")

    # Chosing threshold
    threshold = thresholds[stakes]

    # Check and transform the format of the data if needed
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        else:
            X = pd.DataFrame.from_dict(X)

    # Rename columns
    X.columns = X.columns.str.replace('\n',' ')
    X.columns = X.columns.str.replace(' ','_')

    # Saving name
    player_name = X.Player_Name

    # Drop useless columns
    for col in ['Player_Name', 'Site', "Hands", "Net_Won"]:
        if col in X.columns:
            X.drop([col], axis=1, inplace=True)

    # Add polynomials and interactions to the dataframe
    extra_features = [
        'WTSD%_*_Won_$_at_SD',
        'Won_$_at_SD_*_River_CBet%',
        'Won_$_at_SD_*_Raise_Two_Raisers',
        'Won_$_at_SD_*_vs_4Bet_Call',
        'VP$IP_*_Flop_CBet%',
        'VP$IP_*_Fold_to_River_CBet',
        'PFR_*_River_CBet%',
        'PFR_*_Fold_to_Turn_CBet',
        'Squeeze_^2',
        'Squeeze_^3',
        'Squeeze_^4',
        'Postflop_Agg%_^2',
        'Postflop_Agg%_^3',
        'Postflop_Agg%_^4',
        'Won_$_at_SD_^2',
        'Won_$_at_SD_^3',
        'Won_$_at_SD_^4',
        'Raise_Turn_CBet_^2',
        'PFR_*_Flop_CBet%',
        'PFR_*_Flop_CBet%_^2',
        'PFR_*_Flop_CBet%_^3',
        'VP$IP_*_Won_$_at_SD',
        'VP$IP_*_Won_$_at_SD_^2',
        'VP$IP_*_Won_$_at_SD_^3',
        'Raise_Two_Raisers_^2',
        'Raise_Two_Raisers_^3',
        'Raise_Two_Raisers_^4',
        'PFR_^2',
        'PFR_^3',
        'Fold_to_Turn_CBet_^2',
        'Fold_to_Turn_CBet_^3',
        '3Bet_^2',
        '3Bet_^3'
    ]

    features_order = ['VP$IP',
                    'W$WSF%',
                    'WTSD%',
                    'Flop_CBet%',
                    'Turn_CBet%',
                    'River_CBet%',
                    'Fold_to_Flop_Cbet',
                    'Fold_to_River_CBet',
                    'Raise_Flop_Cbet',
                    'Raise_River_CBet',
                    'Call_Two_Raisers',
                    'vs_3Bet_Fold',
                    'vs_3Bet_Call',
                    'vs_3Bet_Raise',
                    'vs_4Bet_Fold',
                    'vs_4Bet_Call',
                    'vs_4Bet_Raise',
                    'WTSD%_*_Won_$_at_SD',
                    'Won_$_at_SD_*_River_CBet%',
                    'Won_$_at_SD_*_Raise_Two_Raisers',
                    'Won_$_at_SD_*_vs_4Bet_Call',
                    'VP$IP_*_Flop_CBet%',
                    'VP$IP_*_Fold_to_River_CBet',
                    'PFR_*_River_CBet%',
                    'PFR_*_Fold_to_Turn_CBet',
                    'Squeeze',
                    'Squeeze_^2',
                    'Squeeze_^3',
                    'Squeeze_^4',
                    'Postflop_Agg%',
                    'Postflop_Agg%_^2',
                    'Postflop_Agg%_^3',
                    'Postflop_Agg%_^4',
                    'Won_$_at_SD',
                    'Won_$_at_SD_^2',
                    'Won_$_at_SD_^3',
                    'Won_$_at_SD_^4',
                    'Raise_Turn_CBet',
                    'Raise_Turn_CBet_^2',
                    'PFR_*_Flop_CBet%',
                    'PFR_*_Flop_CBet%_^2',
                    'PFR_*_Flop_CBet%_^3',
                    'VP$IP_*_Won_$_at_SD',
                    'VP$IP_*_Won_$_at_SD_^2',
                    'VP$IP_*_Won_$_at_SD_^3',
                    'Raise_Two_Raisers',
                    'Raise_Two_Raisers_^2',
                    'Raise_Two_Raisers_^3',
                    'Raise_Two_Raisers_^4',
                    'PFR',
                    'PFR_^2',
                    'PFR_^3',
                    'Fold_to_Turn_CBet',
                    'Fold_to_Turn_CBet_^2',
                    'Fold_to_Turn_CBet_^3',
                    '3Bet',
                    '3Bet_^2',
                    '3Bet_^3']

    # Recreating interactions features
    for feature in features_order:
        if "*" in feature and "^" not in feature:
            features = feature.split("_*_")
            X[feature] = np.array(X[features[0]]) * np.array(X[features[1]])

    # Recreating polynomials features
    for feature in features_order:
        if feature[-2:-1] == "^":
            feature_to_poly = feature[:-3]
            exp = feature[-1]
            X[feature] = np.array(X[feature_to_poly]) ** int(exp)
    
    X = X[features_order]
    # print(X.iloc[0])
    # print("-"*100)
    # Generate prediction probabilities
    preds_proba = np.array(model.predict_proba(X))[:,1]
    pred_class = preds_proba_to_preds_class(preds_proba, threshold)
    print(f"Player name: {player_name} Prediction probability {preds_proba} ---- Class predicted value: {pred_class}")
    if pred_class:
        return "Winning Player"
    return "Losing Player"