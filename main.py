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
 
    te = TransactionEncoder()
    te_arr = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_arr, columns=te.columns_)

    total_transactions = len(df)
    min_support = minsuppc/total_transactions
    print(f'Total: {total_transactions}, Count: {minsuppc}, Support: {min_support}')

    frequent_itemsets = apriori(df, min_support = min_support)

    return frequent_itemsets

def generate_rules(frequent_itemsets, minconf):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    return rules

def main():
    args = parse_args()
    filename = args.filename
    min_support_count = args.minsuppc
    min_confidence = args.minconf

    data = read_values(filename)

    frequent_itemsets = apriori_algorithm(data, min_support_count)
    print("Frequent Itemsets:\n", frequent_itemsets)

    frequent_rules = generate_rules(frequent_itemsets, min_confidence)
    print("Frequent Rules:\n", frequent_rules)

    return 0

if __name__ == "__main__":
    main()
