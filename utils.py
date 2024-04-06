from pyexpat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

def plot_distribution(data, column, title, xlabel, ylabel, num_bins=30, figsize=(10, 6)):
    """
    This function plots the distribution of a given column from a DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data.
    column (str): The column in the DataFrame to plot.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    num_bins (int): The number of bins to use in the histogram.
    figsize (tuple): The size of the figure (default is (10, 6)).

    Returns:
    None. The function displays the plot but does not return any value.
    """
    # Set the figure size
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the distribution of the specified column
    sns.histplot(data[column], kde=False, ax=ax, color='skyblue', edgecolor='#000c66', bins=num_bins)

    # Remove the left spine for aesthetic purposes
    sns.despine(left=True)

    # Set the style to white for aesthetic purposes
    sns.set_style("white")

    # Set the title of the plot
    ax.set_title(title, fontsize=15)

    # Set the x and y labels of the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Display the plot
    plt.show()
    
def percentage(data):
    """
    This function calculates the percentage of missing values in each column of a DataFrame 
    and plots a bar chart of the results.

    Parameters:
    data (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    None. The function displays the plot but does not return any value.
    """
    # Calculate the percentage of missing values in each column
    missing = data.isnull().sum() * 100 / len(data)

    # Create a DataFrame with the results
    percentage_missing = pd.DataFrame({'column': data.columns,
                                       'missing_percentage %': missing.values})

    # Round the percentages to two decimal places
    percentage_missing['missing_percentage %'] = percentage_missing['missing_percentage %'].round(2)

    # Sort the DataFrame by the percentage of missing values
    percentage_missing.sort_values('missing_percentage %', ascending=False, inplace=True)

    # Reset the index of the DataFrame
    percentage_missing.reset_index(drop=True, inplace=True)

    # Plot the percentage of missing values
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x='missing_percentage %', y='column', data=percentage_missing, color='#E1341E')

    # Annotate the bars with the percentage of missing values
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_width() + '%', 
                    xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(8, 0), 
                    textcoords='offset points', 
                    ha="left", 
                    va="center", 
                    fontsize=10)

    # Set the title and labels of the plot
    plt.title('Missing values Percentage for Each Column', fontsize=17, fontweight='bold')
    plt.ylabel('Column', fontsize=12)
    plt.xlabel('Missing percentage %', fontsize=12)

    # Set the x-axis limit
    plt.xlim(0, 50)

    # Display the plot
    plt.show()
    
def categorize_score(score):
    """
    Categorize GRE scores into four categories: 'Excellent', 'Good', 'Average', and 'Below Average'.
    
    Parameters:
    score (int): The GRE score to categorize.

    Returns:
    str: The category of the score.
    """
    if score >= 325:
        return 'Excellent'
    elif score >= 317:
        return 'Good'
    elif score >= 308:
        return 'Average'
    else:
        return 'Below Average'
    
def plot_binary_feature_count(data, feature):
    """
    Plot a countplot for a binary feature in a DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the feature.
    feature (str): The name of the feature to plot.

    """
    
    # Create a figure for the plot
    plt.figure(figsize=(10,6))
    
    # Create a countplot for the feature
    sns.countplot(x=feature, data=data, palette='viridis', hue=feature, legend=False)
    
    # Remove the top and right spines from the plot
    sns.despine(left=True)
    
    # Set the title, x-label, and y-label of the plot
    plt.title(f'Countplot of {feature.capitalize()}', fontsize=20)
    plt.xlabel(feature.capitalize(),fontsize=15)
    plt.ylabel('Count',fontsize=15)
    
    # Set the font size for the x and y ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Display the plot
    plt.show()
    
def plot_correlation(data, feature, target, hue=None):
    """
    Plot the correlation between a feature and a target variable.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the feature and target.
    feature (str): The name of the feature to plot.
    target (str): The name of the target variable to plot.
    hue (str, optional): The name of the variable to color code.
    """
    # Create a figure for the plot
    plt.figure(figsize=(10, 6))

    # Create a scatterplot for the feature and target
    sns.scatterplot(x=feature, y=target, hue=hue, data=data,
                    palette='coolwarm', edgecolor='black')

    # Remove the top and right spines from the plot
    sns.despine(left=True)

    # Format the feature and target names for the plot title and labels
    feature_name = ' '.join(word.capitalize() for word in feature.split('_'))
    target_name = ' '.join(word.capitalize() for word in target.split('_'))

    # Format the hue name for the legend title
    if hue:
        hue_name = ' '.join(word.capitalize() for word in hue.split('_'))
        plt.legend(title=hue_name)

    # Set the title, x-label, and y-label of the plot
    plt.title(f'Correlation between {feature_name} and {target_name}',
              fontsize=20)
    plt.xlabel(feature_name, fontsize=15)
    plt.ylabel(target_name, fontsize=15)

    # Set the font size for the x and y ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Display the plot
    plt.show()
    
def plot_heatmap(data, title="Heatmap"):
    """
    This function plots a heatmap for the given dataframe.

    Parameters:
    data (pandas.DataFrame): The dataframe to plot.
    title (str): The title of the heatmap. Default is "Heatmap".
    """
    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Plot the heatmap with correlation of dataframe's columns
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

    # Set the title of the heatmap
    plt.title(title, fontsize=20)

    # Set the fontsize of xticks and yticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Display the plot
    plt.show()
    
def plot_countplot(data, x, hue=None, title="Countplot", x_label="X", y_label="Count"):
    """
    This function plots a countplot for the given data using seaborn.

    Parameters:
    data (pandas.DataFrame): The dataframe to plot.
    x (str): The column name to be used for the x-axis.
    hue (str): The column name to be used for color encoding. Default is None.
    title (str): The title of the countplot. Default is "Countplot".
    x_label (str): The label for the x-axis. Default is "X".
    y_label (str): The label for the y-axis. Default is "Count".

    Returns:
    None
    """
    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Calculate the counts and sort them in descending order
    order = data[x].value_counts().sort_values(ascending=False).index

    # Create a color palette with the same number of colors as unique values in the x column
    palette = sns.color_palette("viridis", len(order))
    
    # Create the countplot with 'viridis' palette
    sns.countplot(x=x, data=data, palette=palette, hue=hue, order=order)

    # Remove the left spine for aesthetics
    sns.despine(left=True)

    # Format the hue name for the legend title
    if hue:
        hue_name = ' '.join(word.capitalize() for word in hue.split('_'))
        plt.legend(title=hue_name)
    
    # Set the title, x-label, and y-label
    plt.title(title, fontsize=20)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # Set the fontsize of xticks and yticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Display the plot
    plt.show()
    
def plot_piechart(data, column, title="Pie Chart", figsize=(10, 6)):
    """
    This function plots a pie chart of a given column from a DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data.
    column (str): The column in the DataFrame to plot.
    title (str): The title of the plot. Default is "Pie Chart".
    figsize (tuple): The size of the figure (default is (10, 6)).

    Returns:
    None. The function displays the plot but does not return any value.
    """
    # Calculate the counts of unique values in the column
    counts = data[column].value_counts()

    # Map the labels
    labels = counts.index.map({0: 'No', 1: 'Yes'})
    
    # Explode the largest pie chart
    explode = [0.1 if freq == max(counts) else 0 for freq in counts]

    # Create the pie chart
    plt.figure(figsize=figsize)
    plt.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%',
        explode=explode,
        startangle=140,
        shadow=True,
        colors=plt.cm.Dark2.colors
    )

    # Set the title of the plot
    plt.title(title, fontsize=20)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')  
    
    # Display the plot
    plt.show()
    
def plot_heatmap(data, title="Heatmap"):
    """
    This function plots a heatmap for the given dataframe.

    Parameters:
    data (pandas.DataFrame): The dataframe to plot.
    title (str): The title of the heatmap. Default is "Heatmap".
    """
    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Plot the heatmap with correlation of dataframe's columns
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

    # Set the title of the heatmap
    plt.title(title, fontsize=20)

    # Set the fontsize of xticks and yticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Display the plot
    plt.show()
    
def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title):
    """
    This function plots the training, cross-validation, and test data.

    Parameters:
    x_train, y_train : Training data (features and target)
    x_cv, y_cv : Cross-validation data (features and target)
    x_test, y_test : Test data (features and target)
    title : Title for the plot

    Returns:
    None
    """
    
    # Set the figure size
    plt.rcParams['figure.figsize'] = [10, 8]
    
    # Set the marker size
    plt.rcParams['lines.markersize'] = 10
    
    # Plot training data in red with 'x' marker
    plt.scatter(x_train, y_train, marker='x', color='red', label='training')
    
    # Plot cross-validation data in blue with 'o' marker
    plt.scatter(x_cv, y_cv, marker='o', color='blue', label='cross validation')
    
    # Plot test data in green with '^' marker
    plt.scatter(x_test, y_test, marker='^', color='green', label='test')
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Show the legend
    plt.legend()
    
    # Display the plot
    plt.show()
    
def build_models():
    """
    This function builds and returns a list of three different models. Each model is a Sequential model
    with a different architecture. The models are named 'model1', 'model2', and 'model3' respectively.

    Returns:
        list: A list of Sequential models.
    """
    # Set the seed for random number generation in TensorFlow to ensure reproducibility
    tf.random.set_seed(20)

    # Define the architecture for the first model
    model_1 = Sequential(
        [
            Dense(25, activation='relu'),
            Dense(15, activation='relu'),
            Dense(1, activation='linear')
        ],
        name = 'model1'
    )

    # Define the architecture for the second model
    model_2 = Sequential(
        [
            Dense(20, activation='relu'),
            Dense(12, activation='relu'),
            Dense(12, activation='relu'),
            Dense(20, activation='relu'),
            Dense(1, activation='linear')
        ],
        name = 'model2'
    )

    # Define the architecture for the third model
    model_3 = Sequential(
        [
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(4, activation='relu'),
            Dense(12, activation='relu'),
            Dense(1, activation='linear')
        ],
        name = 'model3'
    )

    # Combine all models into a list
    model_list = [model_1, model_2, model_3]

    # Return the list of models
    return model_list