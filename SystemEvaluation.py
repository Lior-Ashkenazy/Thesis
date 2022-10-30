import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt


# ----Calculate Quality----
# Quality = successes/attempts
def get_quality(labels_test, labels_attention_ALL):
    quality_ALL = []
    for labels_attention in labels_attention_ALL:
      sum_attempts = 0
      sum_successes = 0
      for (label_test, label_attention) in zip(labels_test, labels_attention):  # running on both arrays in parallel
          sum_attempts = sum_attempts + 1
          if label_test == label_attention:
              sum_successes = sum_successes + 1
      quality = sum_successes/sum_attempts
      quality_ALL.append(quality)
    print(quality_ALL)
    average_quality = sum(quality_ALL)/len(quality_ALL)
    print(average_quality)
    return average_quality

# ----Probabilities -> Predictions----
def predictions(probabilities):
    threshold = 0.5
    preds = np.where(probabilities[:, 2] > threshold, 1, 0)
    return preds

# ----NLP Metrics----
# F1 Score
def f1Score(y_true,y_pred):
    f1_score_ = f1_score(y_true, y_pred, average='macro')
    return round(f1_score_,4)

# Precision
def precision(y_true,y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    return precision

def recall(y_true,y_pred):
    recall = recall_score(y_true, y_pred, average='macro')
    return recall

# Accuracy
def accuracy(y_true,y_pred):
    acc = accuracy_score(y_true, y_pred)
    return round(acc,4)

# ----Plots----
# bar plots for accuracy and F1 score
def barPlot(X_label, X_values, Y_label, Y_values, title):
    # function to add value labels
    def addlabels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha='center', fontsize=12)

    # creating data on which bar chart will be plot
    x = X_values
    y = Y_values

    # setting figure size by using figure() function
    plt.figure(figsize=(8, 5))

    # making the bar chart on the data
    plt.bar(x, y, color=['#aee2f2', '#62a7bd', '#186d87', '#083e4d'])

    # calling the function to add value labels
    addlabels(x, y)

    # giving title to the plot
    plt.title(title, fontsize=20)

    # giving X and Y labels
    plt.xlabel(X_label, fontsize=14)
    plt.ylabel(Y_label, fontsize=14)

    # visualizing the plot
    plt.show()

def quality_plot(X_labels, Y_labels, title, quality_dataset1, quality_dataset2, quality_dataset3,
                 dataset1, dataset2, dataset3):
    x = np.arange(len(X_labels))  # the label locations
    y = np.arange(len(Y_labels))
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 5))
    rects1 = ax.bar(x + width, quality_dataset1, width, label= dataset1 + 'Data', color='#edea8e')
    rects2 = ax.bar(x + width*2, quality_dataset2, width, label= dataset2 + 'Data', color='#bab757')
    rects3 = ax.bar(x + width*3, quality_dataset3, width, label= dataset3 + 'Data', color='#f5f4ce')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Quality', fontsize=16)
    ax.set_xlabel('Fixed average attention size', fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.set_xticks(x + width*2, X_labels, fontsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)

    ax.bar_label(rects1, padding=3, fontsize=9, label_type='edge')
    ax.bar_label(rects2, padding=3, fontsize=9, label_type='center')
    ax.bar_label(rects3, padding=3, fontsize=9, label_type='edge')

    fig.tight_layout()

    plt.show()

plot_RF = quality_plot([0.9605, 0.9651, 0.86, 0.9507, 0.9547], [0.8747, 0.8960, 0.9173, 0.9240, 0.9307], "Random Forest")


# triangle functions plot
def plot_triangle(c_arr, d_arr, title):
    def plot(c, d):
        a = [0, 0]
        b = [1, 0]
        x1, x2 = np.linspace(0, c, 100), np.linspace(c, 1, 100)
        m1, m2 = (a[1] - d) / (a[0] - c), (b[1] - d) / (b[0] - c)
        n1, n2 = a[1] - m1 * a[0], b[1] - m2 * b[0]
        y1, y2 = m1 * x1 + n1, m2 * x2 + n2
        arr = [x1, y1, x2, y2]
        return arr

    color_arr = ['blue', 'red']
    label_arr = ['Kaggle Data', 'Snapshot Data']
    for c, d, color, label in zip(c_arr, d_arr, color_arr, label_arr):
        triangle_values = plot(c,d)
        plt.plot(triangle_values[0], triangle_values[1], '-r', color=color, label=label)
        plt.plot(triangle_values[2], triangle_values[3], '-r', color = color)
        plt.text(c, d, '  (c,d) = ' + '(' + str(c) + ',' + str(d) + ')')
    plt.xlabel('Acceptance Probability ', color='#1C2833', fontsize = 14)
    plt.ylabel('Attention Size', color='#1C2833', fontsize = 14)
    plt.title(title, fontsize = 20)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.show()