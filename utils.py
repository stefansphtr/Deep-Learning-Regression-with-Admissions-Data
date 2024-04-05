import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
def percentage(dataframe):
    """
    This function calculates the percentage of missing values in each column of a DataFrame 
    and plots a bar chart of the results.

    Parameters:
    dataframe (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    None. The function displays the plot but does not return any value.
    """
    # Calculate the percentage of missing values in each column
    missing = dataframe.isnull().sum() * 100 / len(dataframe)

    # Create a DataFrame with the results
    percentage_missing = pd.DataFrame({'column': dataframe.columns,
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
    
def plot_binary_feature_count(df, feature):
    """
    Plot a countplot for a binary feature in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the feature.
    feature (str): The name of the feature to plot.

    """
    
    # Create a figure for the plot
    plt.figure(figsize=(10,6))
    
    # Create a countplot for the feature
    sns.countplot(x=feature, data=df, palette='viridis', hue=feature, legend=False)
    
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
    
def plot_correlation(df, feature, target, hue=None):
    """
    Plot the correlation between a feature and a target variable.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the feature and target.
    feature (str): The name of the feature to plot.
    target (str): The name of the target variable to plot.
    hue (str, optional): The name of the variable to color code.
    """
    # Create a figure for the plot
    plt.figure(figsize=(10, 6))

    # Create a scatterplot for the feature and target
    sns.scatterplot(x=feature, y=target, hue=hue, data=df,
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
    
def plot_heatmap(dataframe, title="Heatmap"):
    """
    This function plots a heatmap for the given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to plot.
    title (str): The title of the heatmap. Default is "Heatmap".
    """
    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Plot the heatmap with correlation of dataframe's columns
    sns.heatmap(dataframe.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

    # Set the title of the heatmap
    plt.title(title, fontsize=20)

    # Set the fontsize of xticks and yticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Display the plot
    plt.show()