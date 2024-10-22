import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

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

def main():
    filename = 'small.txt'
    data = read_values(filename)

    te = TransactionEncoder()
    te_arr = te.fit(data).transform(data)
    df = pd.DataFrame(te_arr, columns=te.columns_)

    total_transactions = len(df)
    min_support_count = 100
    min_support = min_support_count/total_transactions

    print(f'Total: {total_transactions}, Count: {min_support_count}, Support: {min_support}')

    frequent_itemsets = apriori(df, min_support = min_support)
    print(frequent_itemsets)

    return 0

if __name__ == "__main__":
    main()
