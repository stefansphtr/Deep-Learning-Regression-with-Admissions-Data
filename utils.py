import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(data, column, title, xlabel, ylabel, figsize=(10, 6)):
    """
    This function plots the distribution of a given column from a DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data.
    column (str): The column in the DataFrame to plot.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    figsize (tuple): The size of the figure (default is (10, 6)).

    Returns:
    None. The function displays the plot but does not return any value.
    """
    # Set the figure size
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the distribution of the specified column
    sns.histplot(data[column], kde=False, ax=ax, color='skyblue', edgecolor='black', bins=30)

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
    
def percentage(orders_df):
    missing = orders_df.isnull().sum()*100 / len(orders_df)

    percentage_missing = pd.DataFrame({'column':orders_df.columns,
                                       'missing_percentage %':missing.values})
    percentage_missing['missing_percentage %'] = percentage_missing['missing_percentage %'].round(2)
    percentage_missing = percentage_missing.sort_values('missing_percentage %', ascending=False)
    percentage_missing = percentage_missing.reset_index()
    percentage_missing = percentage_missing.drop('index', axis=1)

    # plot the missing value percentage
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x='missing_percentage %', y='column', data=percentage_missing, color='#E1341E')
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_width() + '%', xy=(p.get_width(), p.get_y()+p.get_height()/2),
                xytext=(8, 0), textcoords='offset points' ,ha="left", va="center", fontsize=10)
    plt.title('Missing values Percentage for Each Column', fontsize=17, fontweight='bold')
    plt.ylabel('Column', fontsize=12)
    plt.xlabel('Missing percentage %', fontsize=12)
    plt.xlim(0,50)
    plt.show()