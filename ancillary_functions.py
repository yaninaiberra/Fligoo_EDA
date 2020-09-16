import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import collections
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def my_histogram(var, color):
    """
    Function that generates a histogram.
    
    :param var: variables to chart
    :param color: color of the histogram bars
    
    :return: N/A
    """
    plt.figure()
    name = var.name
    plt.hist(var, color=color, alpha=0.5, bins=20)
    plt.xlabel(name)
    plt.ylabel('Events')
    plt.title(f'{name} Variable Distribution')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()
    

def outlier_days_past_due_not_worse(df, attribute):
    """
    Function to handle outliers of a specific feature.
    
    :param df: dataframe for outlier delete
    :param attribute: feature with outlier
    
    :return: N/A
    """
    New = []
    med = df[attribute].median()
    for val in df[attribute]:
        if ((val == 98) | (val == 96)):
            New.append(med)
        else:
            New.append(val)
    df[attribute] = New
    

def detect_outliers(df,n,features):
    """
    Function to detect outliers of a specific feature.
    
    :param df: dataframe for outlier delete
    :param n: value for comparisson
    :param attribute: feature with outlier
    
    :return: multiple_outliers list
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1        
        # outlier step
        outlier_step = 1.5 * IQR        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)        
    # select observations containing more than 2 outliers
    outlier_indices = collections.Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )    
    return multiple_outliers
    
def my_zscore(df, attribute, threshold):
    """
    Function to manage outliers with z-score function.
    
    :param df: dataframe for outlier delete
    :param attribute: feature with outlier
    :param threshold: value for threshold comparisson
    
    :return: multiple_outliers list
    """
    z = stats.zscore(np.array(df[attribute]))
    z_index = df[attribute][np.abs(z) < threshold].index
    return df.loc[z_index]
    
    
def mad_based_outlier(points, threshold=3.5):
    """
    Function to manage outliers with z-score and other statitics functions.
    
    :param points: feature for outlier delete
    :param threshold: value for threshold comparisson
    
    :return: 0 or 1
    """
    median_y = np.median(points)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in points])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in points]
    return np.abs(modified_z_scores) > threshold


def evaluation(y_test, predictions):
    """
    Function that returns the mean absolute error and the mean percentage absolute error.
    
    :param y_test: real values
    :param predictions: predicts values
    
    :return: Returns the mean absolute error and the mean absolute percentage error.
    """
    # Calcula mae 
    mae = mean_absolute_error(y_test, predictions)
    #calculate mape 
    mape = np.mean((np.abs(y_test - predictions)/y_test)*100)
    #print calculated values
    print(f"El error absoluto medio para el modelo es {round(mae, 2)}")
    print(f"El error porcentual absoluto medio para el modelo es {round(mape, 2)}")
    

def graph_real_pred(y_test, predictions, color):
    """
    Function that graphs the real vs. predicted values.
    :param y_test: real values
    :param predictions: predicts values
    :param color: plot color
    
    :return: Scatterplot showing the relationship between actual and predicted value
    """
    plt.scatter(y_test, predictions, c=color, s=10)    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Real', size=15, labelpad=1)
    plt.ylabel('Predicted', size=15, labelpad=1)
    plt.show()


def feature_importance(model, feature_list):
    """
    Function that gets and plots the feature importance
    for the given model
    :param model: the model to evaluaate
    :param feature_list: a list of features contained in the model

    :returns a plot with feature importance
    """
    # Obtiene la lista de importancias 
    importances = list(model.feature_importances_)
    # Junta los nombres de los atributos y las importancias
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Ordena por orden de importancia
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print la lista de importancias
    [print('Variable: {} Importance: {}'.format(*pair)) for pair in feature_importances];
    # Colores
    colors = cm.rainbow(np.linspace(0, 1, len(feature_list)))
    
    # Caracteristicas en orden de importancia 
    characteristics = [x[0] for x in feature_importances]
    # Obtiene las importancias
    importances_plot = [x[1] for x in feature_importances]
    # Grafica un bar plot
    plt.bar(characteristics, importances_plot, color=colors)
    # Personalizamos el grafico
    plt.xticks(list(range(len(characteristics))), characteristics, rotation = 90)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gcf().subplots_adjust(bottom=0.3);

