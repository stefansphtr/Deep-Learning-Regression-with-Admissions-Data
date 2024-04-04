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
    sns.histplot(data[column], kde=False, ax=ax, color='skyblue', edgecolor='black', bins=num_bins)

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
    if score >= 162:
        return 'Excellent'
    elif score >= 152:
        return 'Good'
    elif score >= 147:
        return 'Average'
    else:
        return 'Below Average'