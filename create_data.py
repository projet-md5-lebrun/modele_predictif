import csv
import random
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from numpy import array

# Set up Spark configuration and context
conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf=conf)

# Define possible values for categorical features
employed_values = ['Y', 'N']
education_values = ['BS', 'MS', 'PhD']
top_tier_values = ['Y', 'N']
interned_values = ['Y', 'N']
hired_values = ['Y', 'N']

# Function to generate random synthetic data for a row based on specified rules
def generate_random_row():
    years_experience = random.randint(0, 30)  # Random years of experience between 0 and 30
    employed = random.choice(employed_values)
    previous_employers = random.randint(0, 10)  # Random number of previous employers between 0 and 10
    
    # Determine education level based on rules
    if years_experience < 3:
        education_level = random.choice(['MS', 'PhD'])  # If years_experience < 3, exclude 'BS'
    else:
        education_level = random.choice(education_values)  # Otherwise, include all education levels
    
    # Determine if hired based on education level
    if education_level == 'BS':
        hired = 'N'  # Never hired for 'BS'
    elif education_level == 'MS':
        hired = 'N' if years_experience < 3 else random.choice(hired_values)  # Never hired for 'MS' if years_experience < 3
    else:
        hired = random.choice(hired_values)  # For 'PhD', random hire decision
    
    top_tier_school = random.choice(top_tier_values)
    interned = random.choice(interned_values)
    
    return [years_experience, employed, previous_employers, education_level, top_tier_school, interned, hired]

# Define CSV file path
csv_file_path = "synthetic_data_with_rules.csv"

# Generate synthetic data and write to CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header row
    writer.writerow(['Years Experience', 'Employed?', 'Previous employers', 'Level of Education', 'Top-tier school', 'Interned', 'Hired'])
    
    # Write 1000 rows of synthetic data
    for _ in range(1000):
        row = generate_random_row()
        writer.writerow(row)

print(f"CSV file with synthetic data generated successfully: {csv_file_path}")

# Load synthetic data from CSV and train DecisionTree classifier
data = sc.textFile(csv_file_path).map(lambda line: line.split(","))
header = data.first()
data = data.filter(lambda row: row != header)

# Function to convert CSV row to LabeledPoint
def parseCsvRow(row):
    years_experience, employed, previous_employers, education_level, top_tier_school, interned, hired = row
    features = [
        int(years_experience),
        1 if employed == 'Y' else 0,
        int(previous_employers),
        1 if education_level == 'BS' else (2 if education_level == 'MS' else 3),
        1 if top_tier_school == 'Y' else 0,
        1 if interned == 'Y' else 0
    ]
    label = 1 if hired == 'Y' else 0
    return LabeledPoint(label, array(features))

# Convert data to LabeledPoints
parsedData = data.map(parseCsvRow)

# Train DecisionTree classifier
model = DecisionTree.trainClassifier(parsedData, numClasses=2,
                                     categoricalFeaturesInfo={1: 2, 3: 4, 4: 2, 5: 2},
                                     impurity='gini', maxDepth=5, maxBins=32)

# Define test candidate based on specified rules
testCandidate = [1, 'N', 1, 'BS', 'N', 'N']

# Convert test candidate to LabeledPoint and predict
labeled_point = parseCsvRow(testCandidate)
prediction = model.predict(labeled_point.features)

# Print prediction result based on rules
print(f"\nTest Candidate - Features: {testCandidate} - Prediction: {'Hired' if prediction == 1.0 else 'Not Hired'}")

# Stop Spark context
sc.stop()
