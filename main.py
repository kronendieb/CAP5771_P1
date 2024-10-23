from functools import total_ordering
from mlxtend import frequent_patterns
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import argparse
import time

# Parse the arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description="Apriori algorithm for transaction data.")

    parser.add_argument('filename', type=str, help="Path to the file of transaction data.")
    parser.add_argument('minsuppc', type=int, default=200, help="Minimum support count, should be less than the number of transactions")
    parser.add_argument('minconf', type=float, default=0.5, help="Minimum confidence, -1 does not generate rules")

    return parser.parse_args()

# Read the values from a file
def read_values(filename):
    result = []
    with open(filename, 'r') as file:
        for line in file:
            if line:
                index, val = map(int, line.split(" "))
                while len(result) <=index:
                    result.append([])
                result[index].append(val)
        return result


def apriori_algorithm(transactions, minsuppc):
 
    total_transactions = len(transactions)
    min_support = minsuppc/total_transactions
    print(f'Total: {total_transactions}, Count: {minsuppc}, Support: {min_support}')

    frequent_itemsets = apriori(transactions, min_support = min_support)

    return frequent_itemsets

def generate_rules(frequent_itemsets, minconf):
    if minconf < 0:
        return
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    return rules

def frequent_itemsets_to_file(frequent_itemsets, transactions, output_file):
    total_transactions = len(transactions)

    with open(output_file, 'w') as file:
        for _, row in frequent_itemsets.iterrows():
            itemset = " ".join(map(str, row['itemsets']))
            support_count = int(row['support'] * total_transactions)
            file.write(f"{itemset}|{support_count}\n")

def frequent_rules_to_file(frequent_rules, transactions, output_file):
    if frequent_rules is None:
        return
    total_transactions = len(transactions)

    with open(output_file, 'w') as file:
        for _, row in frequent_rules.iterrows():
            lhs = " ".join(map(str,row['antecedents']))
            rhs = " ".join(map(str,row['consequents']))
            support_count = int(row['support'] * total_transactions)
            confidence = int(row['confidence'])
            file.write(f"{lhs}|{rhs}|{support_count}|{confidence}\n")

def get_longest_transaction(transactions):
    lengths = transactions.sum(axis=1)
    longest = lengths.max()
    return longest

def get_itemset_lengths(frequent_itemsets):
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    itemsets_by_size = frequent_itemsets['length'].value_counts().sort_index()
    return itemsets_by_size

def get_rule_with_highest_confidence(frequent_rules):
    return frequent_rules.loc[frequent_rules['confidence'].idxmax()]

def info_to_file(args, transactions, frequent_itemsets, frequent_rules, times, output_file):
    with open(output_file, 'w') as file:
        total_items = len(transactions.columns)
        total_transactions = len(transactions)
        longest_transaction = get_longest_transaction(transactions)
        itemsets_by_size = get_itemset_lengths(frequent_itemsets)
        total_frequent_itemsets = len(frequent_itemsets)
        total_frequent_rules = len(frequent_rules)
        # Highest Confidence Rule
        hcr = get_rule_with_highest_confidence(frequent_rules)

        # Output the args
        file.write(f"minsuppc:{args.minsuppc}\nminconf:{args.minconf}\ninput file:{args.filename}\n")
        # Output the transaction totals:
        file.write(f"Number of items:{total_items}\nNumber of transactions:{total_transactions}\nThe length of the longest transaction:{longest_transaction}\n")

        # Output itemeset values
        for i, val in enumerate(itemsets_by_size):
            file.write(f"Number of frequent {i+1}-itemsets:{val}\n")
        file.write(f"Total number of frequent itemsets:{total_frequent_itemsets}\n")

        # Output rules values:
        file.write(f"Number of high-confidence rules:{total_frequent_rules}\n")
        file.write(f"The rule with the highest confidence:{' '.join(map(str,hcr['antecedents']))}|{' '.join(map(str,hcr['consequents']))}|{int(hcr['support']*total_transactions)}|{hcr['confidence']:.4f}\n")
        file.write(f"Time in seconds to find the frequent itemsets:{times[0]}\nTime in seconds to find the confident rules:{times[1]}")

def main():
    args = parse_args()
    filename = args.filename
    min_support_count = args.minsuppc
    min_confidence = args.minconf

    data = read_values(filename)
    te = TransactionEncoder()
    te_arr = te.fit(data).transform(data)
    df = pd.DataFrame(te_arr, columns=te.columns_)

    times = [0.0, 0.0]

    print("Finding frequent itemsets...")
    start_time = time.time()
    frequent_itemsets = apriori_algorithm(df, min_support_count)
    end_time = time.time()
    times[0] = end_time - start_time
    print(f"Done! Time taken to find frequent itemsets: {times[0]:.4f} seconds")

    print("Finding rules with high confidence...")
    start_time = time.time()
    frequent_rules = generate_rules(frequent_itemsets, min_confidence)
    end_time = time.time()
    times[1] = end_time - start_time
    print(f"Done! Time taken to find frequent itemsets: {times[1]:.4f} seconds")

    frequent_itemsets_to_file(frequent_itemsets, df, "items_01.txt")
    frequent_rules_to_file(frequent_rules, df, "rules_01.txt")
    info_to_file(args, df, frequent_itemsets, frequent_rules, times, "info_01.txt")

    return 0

if __name__ == "__main__":
    main()
