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
    # List of transactions
    result = []

    # Open the input file to read the list of items.
    with open(filename, 'r') as file:
        # For each line in the file read it into the correct transaction in the list.
        for line in file:
            # If the line exists
            if line:
                # Get the index of the transaction, and the value of the item to be appended
                index, val = map(int, line.split(" "))
                # If the index transaction does not exist create a transaction until it does.
                while len(result) <=index:
                    # Append an empty transaction to the list.
                    result.append([])
                # Append the value of the item to the correct transaction.
                result[index].append(val)
        # Return the list of transactions.
        return result


# Creates the frequent itemsets from transaction and support count.
def apriori_algorithm(transactions, minsuppc):
    # If minimum support is not within boundaries do nothing.
    if minsuppc < 0:
        print("ERROR: Minimum support count must be greater than 0.")
        return
 
    # Get the total number of transaction
    total_transactions = len(transactions)
    # Use the total number of transactions to transform support count to support value.
    min_support = minsuppc/total_transactions

    # Get the frequent itemsets using the apriori algorithm from mlxtend
    frequent_itemsets = apriori(transactions, min_support = min_support, use_colnames=True)

    return frequent_itemsets

# Generates the rules from the frequent itemsets and minimum confidence
def generate_rules(frequent_itemsets, minconf):
    # Do not generate rules if minconf is not within confidence boundary
    if minconf < 0:
        return

    # Create the rules using mlxtend
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    return rules

# Print the frequent itemsets to a file
def frequent_itemsets_to_file(frequent_itemsets, transactions, output_file):
    # Total number of transactions, to find support count.
    total_transactions = len(transactions)

    # Open the itemsets file to print into
    with open(output_file, 'w') as file:

        # For each row, print the information into the file
        for _, row in frequent_itemsets.iterrows():
            # Create a string of itemsets separated by a space
            itemset = " ".join(map(str, row['itemsets']))
            # Get the support count from the suppor value times transaction total.
            support_count = int(row['support'] * total_transactions)
            # Print the information into the file
            file.write(f"{itemset}|{support_count}\n")

# Print the high confidence rules into a file
def frequent_rules_to_file(frequent_rules, transactions, output_file):
    # If no rules were generated, do nothing.
    if frequent_rules is None:
        return

    # Total number of transactions, to find support count.
    total_transactions = len(transactions)

    # Open the output file for rules
    with open(output_file, 'w') as file:
        # For each rule print the information into the file
        for _, row in frequent_rules.iterrows():
            # Get the left hand side by antecedents
            lhs = " ".join(map(str,row['antecedents']))
            # Get the right hand side by consequents
            rhs = " ".join(map(str,row['consequents']))
            # Get the support count from the support value
            support_count = int(row['support'] * total_transactions)
            # Get the confidence from the transaction 
            confidence = float(row['confidence'])
            # Print the information into the file
            file.write(f"{lhs}|{rhs}|{support_count}|{confidence}\n")

# Get the longest transaction from the transaction list
def get_longest_transaction(transactions):
    # for each transaction, get the number of items then return the max of those lengths.
    lengths = transactions.sum(axis=1)
    longest = lengths.max()
    return longest

# Get the Length of each itemset
def get_itemset_lengths(frequent_itemsets):
    # For each itemset give it a size value in a new column "length"
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    # Sort the itemsets by the size of the length given previously
    itemsets_by_size = frequent_itemsets['length'].value_counts().sort_index()
    return itemsets_by_size

# Get the rule with the highest confidence
def get_rule_with_highest_confidence(frequent_rules):
    # Return the rule with the highest confidence
    return frequent_rules.loc[frequent_rules['confidence'].idxmax()]

# Print the required information to a file.
def info_to_file(args, transactions, frequent_itemsets, frequent_rules, times, output_file):
    # Open the output file to print information
    with open(output_file, 'w') as file:
        # Get the total items in the transaction dataframe
        total_items = len(transactions.columns)
        # Get the total number of transactions
        total_transactions = len(transactions)
        # Find the longest transaction by number of items
        longest_transaction = get_longest_transaction(transactions)
        # Sort a list of itemsets by number of items
        itemsets_by_size = get_itemset_lengths(frequent_itemsets)
        # Find the total number of frequent itemsets
        total_frequent_itemsets = len(frequent_itemsets)
        # Find the total number of high confidence rules
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
        file.write(f"The rule with the highest confidence:{' '.join(map(str,hcr['antecedents']))}|{' '.join(map(str,hcr['consequents']))}|{int(hcr['support']*total_transactions)}|{float(hcr['confidence'])}\n")
        file.write(f"Time in seconds to find the frequent itemsets:{times[0]}\nTime in seconds to find the confident rules:{times[1]}")

def main():
    # We obtain and parse the arguments passed into the program thgough the command line
    args = parse_args()
    filename = args.filename
    min_support_count = args.minsuppc
    min_confidence = args.minconf

    # The following are the names of the output files
    items_filename = "items01.txt"
    rules_filename = "rules01.txt"
    info_filename = "info01.txt"

    # We read the data from the read_values function from the input file
    data = read_values(filename)
    # Encode the data from the input file into a transaction dataframe
    te = TransactionEncoder()
    te_arr = te.fit(data).transform(data)
    df = pd.DataFrame(te_arr, columns=te.columns_)

    # List to take track of time
    times = [0.0, 0.0]

    # Start finding the frequent itemsets.
    print("Finding frequent itemsets...")
    start_time = time.time() # Time at the start point of the function.
    # Find the frequent itemsets
    frequent_itemsets = apriori_algorithm(df, min_support_count)
    end_time = time.time() # Time at the end of the function
    times[0] = end_time - start_time # Elapsed time
    print(f"Done! Time taken to find frequent itemsets: {times[0]:.4f} seconds")
    print(frequent_itemsets)

    print("Finding rules with high confidence...")
    start_time = time.time() # Time at the start point of the function
    # Generate rules from the frequent itemset given a confidence
    frequent_rules = generate_rules(frequent_itemsets, min_confidence)
    end_time = time.time() # Time at the end of the function
    times[1] = end_time - start_time # Elapsed time
    print(f"Done! Time taken to find frequent rules: {times[1]:.4f} seconds")
    print(frequent_rules)

    # Print the itemsets to file
    frequent_itemsets_to_file(frequent_itemsets, df, items_filename)
    # Print the rules to file
    frequent_rules_to_file(frequent_rules, df, rules_filename)
    # Print the program information to file
    info_to_file(args, df, frequent_itemsets, frequent_rules, times, info_filename)

    return 0

if __name__ == "__main__":
    main()
