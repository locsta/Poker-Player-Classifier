import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc

def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
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

def print_metrics(self, labels, preds):
        print("Precision Score: {}".format(precision_score(labels, preds)))
        print("Recall Score: {}".format(recall_score(labels, preds)))
        print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
        print("F1 Score: {}".format(f1_score(labels, preds)))


def lr_roc(y_test, y_hat_test, plot=False):
    if not plot:
        return auc(test_fpr, test_tpr)
    else:
        test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_hat_test)
        print('AUC: {}'.format(auc(test_fpr, test_tpr)))
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
        return auc(test_fpr, test_tpr)
    

def print_corr(df, pct=0):
    
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

# class predictions():
#     def __init__(self):
#         self.predictions = "No predictions, you need to run a prediction Method first"
#         self.model = "No model chosen yet"

#     def print_metrics(self, labels=self.labels, preds=self.preds):
#         print("Precision Score: {}".format(precision_score(labels, preds)))
#         print("Recall Score: {}".format(recall_score(labels, preds)))
#         print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
#         print("F1 Score: {}".format(f1_score(labels, preds)))

#     def KNN(self, X=X_train, verbose=True):
#         pass
#         if verbose:
#             self.print_metrics(labels, preds) #use self.params

#     def logistic_reg(self, X=X_train, verbose=True):
#         pass

#     def decision_trees(self, X=X_train, verbose=True):
#         pass