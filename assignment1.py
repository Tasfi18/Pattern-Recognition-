import csv
import random
import math

def load_csv(filename):
    data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                nums = [float(x) for x in row[:-1]]
                label = row[-1]
                data.append((nums, label))
    return data

def split_data(data, ratio):
    random.shuffle(data)
    cut = int(len(data) * ratio)
    return data[:cut], data[cut:]

def group_by_class(data):
    groups = {}
    for features, label in data:
        if label not in groups:
            groups[label] = []
        groups[label].append(features)
    return groups

def prior_probabilities(data):
    total = len(data)
    grouped = group_by_class(data)
    priors = {}
    for label, rows in grouped.items():
        priors[label] = len(rows) / total
    return priors

def average(values):
    return sum(values) / len(values)

def stdev(values):
    m = average(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)

def information_feature(data):
    return [(average(col), stdev(col)) for col in zip(*data)]

def information_classes(data):
    grouped = group_by_class(data)
    classes = {}
    for label, rows in grouped.items():
        classes[label] = information_feature(rows)
    return classes

def gaussian_probability(x, mean, sd):
    if sd == 0:
        return 1 if x == mean else 0
    exp = math.exp(-((x - mean) ** 2) / (2 * sd ** 2))
    return (1 / (math.sqrt(2 * math.pi) * sd)) * exp

def class_probability(summery_info, row, priors):
    probability = {}
    for label, features in summery_info.items():
        probability[label] = priors[label]
        for i in range(len(features)):
            mean, sd = features[i]
            x = row[i]
            probability[label] *= gaussian_probability(x, mean, sd)
    total = sum(probability.values())
    for key in probability:
        probability[key] = probability[key] / total
    return probability

def predict(summery_info, row, priors):
    probability = class_probability(summery_info, row, priors)
    return max(probability, key=probability.get), probability

def test_model(summery_info, test, priors):
    right = 0
    for row, label in test:
        prediction, probability = predict(summery_info, row, priors)
        print("Input:", row)
        print("Class probabilities:")
        for cls, prob in probability.items():
            print("   ", cls, ":", round(prob, 4))
        print("Predicted:", prediction, " Actual:", label)
        if prediction == label:
            print("Correct")
            right += 1
        else:
            print("Wrong")
        print("\n ")
        if prediction == label:
            right += 1
    print("Accuracy:", (right / len(test)) * 100, "%")

if __name__ == "__main__":
    file = r"C:\Users\ADMIN\Desktop\pattern recognition\iris.csv"
    data = load_csv(file)
    ratio = float(input("Training ratio: "))
    train, test = split_data(data, ratio)
    summery_info = information_classes(train)
    priors = prior_probabilities(train)

    print("\nTesting ")
    test_model(summery_info, test, priors)