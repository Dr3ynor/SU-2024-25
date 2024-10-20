import itertools
from itertools import combinations

# --- 1 ---
data = [1,2,3,4,5,6]


result = {}
counter = 0

# 3 for loops
for i in range(0, len(data)):
    for j in range(i+1, len(data)):
        for k in range(j+1, len(data)):
            result[counter] = [data[i], data[j], data[k]]
            counter = counter + 1

print("-------- 1) RECURSION --------")
result = {}
counter = 0

def generate_combinations(data, combination_length, start=0, current_combination=None, result=None):
    if current_combination is None:
        current_combination = []
    if result is None:
        result = []

    if len(current_combination) == combination_length:
        result.append(current_combination[:])
        return result

    for i in range(start, len(data)):
        current_combination.append(data[i])
        generate_combinations(data, combination_length, i + 1, current_combination, result)
        current_combination.pop()

    return result

combinations = generate_combinations(data, 3)
for combination in combinations:
    result[counter] = combination
    counter += 1
print(result)



# --- 2 ---
def read_file_to_list_of_lists(file_path):
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            result.append(line.strip().split())
    return result

file_path = "/home/jakub/school/7th_semester/SU/1_tyden/test.txt"

test_data = read_file_to_list_of_lists(file_path)

def generate_frequent_patterns(data, min_support):
    pattern_counts = {}
    total_transactions = len(data)
    for transaction in data:
        for i in range(1, len(transaction) + 1):
            for combination in itertools.combinations(transaction, i):
                if combination in pattern_counts:
                    pattern_counts[combination] += 1
                else:
                    pattern_counts[combination] = 1


    frequent_patterns = {}
    for pattern, count in pattern_counts.items():
        support = count / total_transactions
        if support >= min_support:
            frequent_patterns[pattern] = support
    return frequent_patterns

min_support = 0.15
frequent_patterns = generate_frequent_patterns(test_data, min_support)
pattern_counts = {}

print("-------- PATTERN + SUPPORT --------")
for pattern, support in frequent_patterns.items():
    print(f"Pattern: {pattern}, Support: {support}")

    pattern_length = len(pattern)
    if pattern_length not in pattern_counts:
        pattern_counts[pattern_length] = 0
    pattern_counts[pattern_length] += 1

print("-------- 3 --------")
for length, count in pattern_counts.items():
    print(f"Length {length}: {count}")

# --- 3 ---
def generate_association_rules(frequent_patterns, min_confidence):
    rules = []
    for pattern in frequent_patterns:
        if len(pattern) > 1:
            for i in range(1, len(pattern)):
                for antecedent in itertools.combinations(pattern, i):
                    consequent = tuple(item for item in pattern if item not in antecedent)
                    antecedent_support = frequent_patterns.get(antecedent, 0)
                    if antecedent_support > 0:
                        confidence = frequent_patterns[pattern] / antecedent_support
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, confidence))
    return rules

min_confidence = 0.5
association_rules = generate_association_rules(frequent_patterns, min_confidence)
for antecedent, consequent, confidence in association_rules:
    print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence:.2f}")
