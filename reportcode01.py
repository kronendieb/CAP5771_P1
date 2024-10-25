import matplotlib.pyplot as plt
from team01 import *

# General function to print a plot of data given an x, y, title, and each label.
def plot_data(x, y, title, subtitle_x, subtitle_y):
    # Plotting the data
    plt.figure(figsize=(8,6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')

    # Adding labels and title
    plt.xlabel(subtitle_x)
    plt.ylabel(subtitle_y)
    plt.title(title)

    # Display the graph
    plt.grid(True)
    plt.show()


# The function the plots each graph
def plot():

    # Read in the data and format it correctly into a dataframe
    data = read_values("small.txt")
    te = TransactionEncoder()
    te_arr = te.fit(data).transform(data)
    df = pd.DataFrame(te_arr, columns=te.columns_)

    # Lists required to plot frequent itemsets
    times = []
    minsuppc = [50, 75, 100, 125, 150, 175, 200] # Required support counts for the report
    num_sets = []

    # For each minimum support count, get the timer and frequent itemset numbers.
    for sup in minsuppc:
        start_time = time.time()
        frequent_itemsets = apriori_algorithm(df, sup) # Run the apriori algorithm to find itemsets.
        end_time = time.time()
        num_sets.append(len(frequent_itemsets)) # Append the number of itemsets into the list
        times.append(end_time - start_time) # Append the time into the list of times

    # Plot the elapsed time for Question 2.
    plot_data(minsuppc, times, "Time Elapsed in Calculation vs Minimum Support Count", "Time Elapsed", "Minimum Support Count")

    # Plot the number of frequent itemsets for Question 3
    plot_data(minsuppc, num_sets, "Number of Frequent Itemsets vs Minimum Support Count", "Number of Frequent Itemsets", "Minimum Support Count")

    # Lists required to plot rule generation
    minconf = [0.7, 0.75, 0.8, 0.85, 0.9]
    min_supp = 100
    num_rules = []

    # For each confidence value run the apriori algorithm and generate rules
    for conf in minconf:
        frequent_itemsets = apriori_algorithm(df, min_supp) # Apriori algorithm
        frequent_rules = generate_rules(frequent_itemsets, conf) # Rule generation.
        num_rules.append(len(frequent_rules)) # Append number of rules generated.

    # Plot the high confidence rules generated for Question 4.
    plot_data(minconf, num_rules, "Number of Rules Created vs Minimum Confidence", "Minimum Confidence", "Number of Rules Created")



if __name__ == "__main__":
    plot()
