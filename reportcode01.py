import matplotlib.pyplot as plt
from team01 import *

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



def plot():

    data = read_values("small.txt")
    te = TransactionEncoder()
    te_arr = te.fit(data).transform(data)
    df = pd.DataFrame(te_arr, columns=te.columns_)

    times = []
    minsuppc = [50, 75, 100, 125, 150, 175, 200]
    num_sets = []
    for sup in minsuppc:
        start_time = time.time()
        frequent_itemsets = apriori_algorithm(df, sup)
        end_time = time.time()
        num_sets.append(len(frequent_itemsets))
        times.append(end_time - start_time)

    plot_data(minsuppc, times, "Time Elapsed in Calculation vs Minimum Support Count", "Time Elapsed", "Minimum Support Count")
    plot_data(minsuppc, num_sets, "Number of Frequent Itemsets vs Minimum Support Count", "Number of Frequent Itemsets", "Minimum Support Count")

    times = []
    minconf = [0.7, 0.75, 0.8, 0.85, 0.9]
    min_supp = 100
    num_rules = []
    for conf in minconf:
        start_time = time.time()
        frequent_itemsets = apriori_algorithm(df, min_supp)
        frequent_rules = generate_rules(frequent_itemsets, conf)
        end_time = time.time()
        num_rules.append(len(frequent_rules))
        times.append(end_time - start_time)

    plot_data(minconf, num_rules, "Number of Rules Created vs Minimum Confidence", "Number of Rules Created", "Minimum Confidence")



if __name__ == "__main__":
    plot()
