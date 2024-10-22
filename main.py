from mlxtend import frequent_patterns
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import argparse

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
    total_transactions = len(transactions)

    with open(output_file, 'w') as file:
        for _, row in frequent_rules.iterrows():
            lhs = " ".join(map(str,row['antecedents']))
            rhs = " ".join(map(str,row['consequents']))
            support_count = int(row['support'] * total_transactions)
            confidence = int(row['confidence'])
            file.write(f"{lhs}|{rhs}|{support_count}|{confidence}\n")

def longest_transaction(transactions):

    lengths = transactions.sum(axis=1)
    longest = lengths.max()
    return longest


def main():
    args = parse_args()
    filename = args.filename
    min_support_count = args.minsuppc
    min_confidence = args.minconf

    data = read_values(filename)
    te = TransactionEncoder()
    te_arr = te.fit(data).transform(data)
    df = pd.DataFrame(te_arr, columns=te.columns_)
    print(longest_transaction(df))


    frequent_itemsets = apriori_algorithm(df, min_support_count)
    print("Frequent Itemsets:\n", frequent_itemsets)

    frequent_rules = generate_rules(frequent_itemsets, min_confidence)
    print("Frequent Rules:\n", frequent_rules)

    frequent_itemsets_to_file(frequent_itemsets, df, "items_01.txt")
    frequent_rules_to_file(frequent_rules, df, "rules_01.txt")

    return 0

if __name__ == "__main__":
    main()
