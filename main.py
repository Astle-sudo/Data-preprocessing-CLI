# Preprocessing Options :
#
# 1. Data Statistics
#  a.Column Statistics (with graphs)
#    : Count
#    : Mean
#    : Median
#    : Mode
#    : Standard Deviation
#    : Quartile Deviations
#    : Kurtosis
#    : Skewness
#    : (Minimum Value, Maximum value)
#  b.Entire Dataset statistics
#    : Column Data types, Plot Graphs
#
# 2. Handle Null Values
#  -Null value counts per column
#  -Drop Columns
#  -Replace
#    : Mean
#    : Mode
#    : Median
#
# 3. Encoding Categorical Data
#  -One Hot Encoding
#
# 4. Feature Scaling
#  -Standardization (Standard Scaler)
#  -Normalization (Minimax scaler)
#
# 5. Split the Data
#  -Split into train and test
#  -Split into train, validation and test
#
# 6. Download the dataset


import sys
import numpy
import pandas
import pandas as pd
import termplotlib as tpl
from scipy import stats
from rich import print as P
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def Vplot(sampleData):
    print()
    print("Graph Plot:")
    counts, bin_edges = numpy.histogram(sampleData, bins=40)
    fig = tpl.figure()
    fig.hist(counts, bin_edges, grid=[15, 25], force_ascii=False)
    fig.show()


def Hplot(sampleData):
    counts, bin_edges = numpy.histogram(sampleData)
    fig = tpl.figure()
    fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
    fig.show()


def mainQuestions():
    print("Pick one of the following preprocessing steps:")
    print()
    print("1. Data Statistics")
    print("2. Handling Null Values")
    print("3. Encoding Categorical Data")
    print("4. Feature Scaling")
    print("5. Split Data")
    print("6. Download processed dataset")
    print()
    userResponse = str(input("Enter an option: "))
    if userResponse == "-2":
        exit()
    if userResponse == "1":
        dataStatistics()
    if userResponse == "2":
        handleNullValues()
    if userResponse == "3":
        encode()
    if userResponse == "4":
        featureScaling()
    if userResponse == "5":
        splitData()
    if userResponse == "6":
        Download()
    if userResponse == "-1":
        mainQuestions()


def dataStatistics():
    print("-------------------------------------------")
    print("Which category ?")
    print()
    print("1. Column Statistics")
    print("2. Dataset Statistics")
    print()
    userResponse = str(input("Enter an option: "))
    if userResponse == "-2":
        exit()
    if userResponse == "1":
        columnStats()
    if userResponse == "2":
        entireDataStats()
    if userResponse == "-1":
        mainQuestions()


def handleNullValues():
    print("-------------------------------------------")
    print()
    print("1. View column-wise Null Values")
    print("2. Drop columns (entire row will be deleted if even one feature is null, or a column with over 75% null values)")
    print("3. Replace with other values")
    print()
    userResponse = str(input("Enter an option: "))
    if userResponse == "-1":
        mainQuestions()
    if userResponse == "-2":
        exit()
    if userResponse == "1":
        viewNullValues()
    if userResponse == "2":
        DropNullValues()
    if userResponse == "3":
        replaceNullValues()


def encode():
    global data
    print()
    print("-------------------------------------------")
    print()
    ob_col = data.select_dtypes(include=['object']).columns
    print(*list(ob_col), sep=", ")
    while True:
        cols = [i for i in input("Write the names of columns to encode: ").split()]
        if set(cols).issubset(set(data.columns)):
            data = pandas.get_dummies(data, columns=cols, dtype=int)
            print(data)
            print()
            break
        else:
            print("Invalid Column name, try again!")
            print()
    mainQuestions()


def viewNullValues():
    global data
    print()
    print(data.isnull().sum())
    print()
    handleNullValues()


def DropNullValues():
    global data
    print()
    data.dropna(inplace=True)
    print(data.isnull().sum())
    print()
    handleNullValues()


def replaceNullValues():
    global data
    print()
    numeric_cols = data.select_dtypes(include=['number']).columns
    print()
    if len(numeric_cols) > 0:
        print(*list(numeric_cols), sep=", ")
        while True:
            column = [str(input("Pick a column: "))]
            if set(column).issubset(set(data.columns)):
                replacement = str(input("Pick mean, median or mode: "))
                if replacement == "mean":
                    data[column] = data[column].replace(numpy.NaN, data[column].mean())
                    print(data[column])
                if replacement == "median":
                    data[column] = data[column].replace(numpy.NaN, data[column].median())
                    print(data[column])
                if replacement == "mode":
                    data[column] = data[column].replace(numpy.NaN, data[column].mode().iloc[0])
                    print(data[column])
                break
            else:
                print("Invalid Column Name, try again!")
                print()
    else:
        print("No columns with Null values")
    print()
    handleNullValues()


def columnStats():
    global data
    numeric_cols = data.select_dtypes(include=['number']).columns
    print()
    print(*list(numeric_cols), sep=", ")
    while True:
        name = [str(input("Pick a column: "))]
        if set(name).issubset(set(data.columns)):
            column = numpy.array(data[name].fillna(value=0))
            if len(column) > 0:
                Vplot(column)
                unique, counts = numpy.unique(column, return_counts=True)
                print()
                P(f'Mean: [yellow]{column.mean()}[/yellow]')
                P(f'Median: [yellow]{numpy.median(column)}[/yellow]')
                P(f'Mode: [yellow]{unique[counts.argmax()]}[/yellow]')
                P(f'Skewness: [yellow]{stats.skew(column)}[/yellow]')
                P(f'Kurtosis: [yellow]{stats.kurtosis(column)}[/yellow]')
                P(f'Standard Deviation: [yellow]{numpy.std(column)}[/yellow]')
                P(f'Inter-Quartile Deviation: [yellow]{numpy.percentile(column, 75) - numpy.percentile(column, 25)}[/yellow]')
                P(f'Range: [yellow]{(column.min(), column.max())}[/yellow]')
                print()
                break
            else:
                print("Error: empty column!")
                print()
                continue
        else:
            print("Invalid Column name, try again!")
            print()
    dataStatistics()


def entireDataStats():
    global data
    print()
    print("Your DataSet:")
    for col_name in data.columns:
        print()
        P(f'{col_name} has data type, [yellow]{data[col_name].dtype}[/yellow]')
        if data[col_name].dtype != "object":
            if not data[col_name].isna().any():
                Hplot(numpy.array(data[col_name]))
            else:
                print(f'{col_name} has NaN values, this cannot be plotted')
        print()
    print()
    dataStatistics()


def featureScaling():
    print()
    print("-------------------------------------------")
    print()
    print("1. Minimax Scaler")
    print("2. Standard Scaler")
    print()
    userResponse = str(input("Enter an option: "))
    if userResponse == "-1":
        mainQuestions()
    if userResponse == "-2":
        exit()
    if userResponse == "1":
        minimax()
    if userResponse == "2":
        standard()


def minimax():
    global data
    print()
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    print(*list(numeric_cols), sep=", ")
    while True:
        cols = [str(i) for i in input("Select columns for scaling: ").split()]
        if set(cols).issubset(set(data.columns)):
            data[cols] = MinMaxScaler().fit_transform(data[cols])
            print()
            print(data[cols])
            break
        else:
            print("Invalid Column name, Try again!")
            print()
    featureScaling()


def standard():
    global data
    print()
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    print(*list(numeric_cols), sep=", ")
    while True:
        cols = [str(i) for i in input("Select columns for scaling: ").split()]
        if set(cols).issubset(set(data.columns)):
            data[cols] = StandardScaler().fit_transform(data[cols])
            print()
            print(data[cols])
            break
        else:
            print("Invalid Column name, Try again!")
            print()
    featureScaling()


def T_V():
    print()
    global data
    print(*list(data.columns), sep=", ")
    cols = [str(i) for i in input("Enter columns to be divided or All for all columns: ").split()]
    ratio = float(input("Enter the ratio (should be a float between 0 and 1): "))
    if "All" in cols:
        train, test = train_test_split(data, test_size=ratio, random_state=42)
        train.to_csv("train.csv", index=False)
        test.to_csv("test.csv", index=False)
    else:
        df = data[cols]
        train, test = train_test_split(df, test_size=ratio, random_state=42)
        train.to_csv("train.csv", index=False)
        test.to_csv("test.csv", index=False)
    print()
    print("Downloaded!")
    splitData()


def T_V_T():
    print()
    global data
    print(*list(data.columns), sep=", ")
    cols = [str(i) for i in input("Enter columns to be divided or All for all columns: ").split()]
    train_other_ratio = float(input("Enter the ratio for train and val/test (should be a float between 0 and 1): "))
    val_test_ratio = float(input("Enter the ratio for validation and test (should be a float between 0 and 1): "))
    if "All" in cols:
        train_df, temp_df = train_test_split(data, test_size=train_other_ratio, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=val_test_ratio, random_state=42)
        train_df.to_csv("train.csv", index=False)
        val_df.to_csv("validation.csv", index=False)
        test_df.to_csv("test.csv", index=False)

    else:
        df = data[cols]
        train_df, temp_df = train_test_split(df, test_size=train_other_ratio, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=val_test_ratio, random_state=42)
        train_df.to_csv("train.csv", index=False)
        val_df.to_csv("validation.csv", index=False)
        test_df.to_csv("test.csv", index=False)
    print()
    print("Downloaded!")
    splitData()


def splitData():
    print()
    print("-------------------------------------------")
    print()
    print("Split your data into training, validation or test datasets:")
    print("The files train.csv, test.csv (validation.csv if selected) will be downloaded")
    print()
    print("1. Train and validation")
    print("2. Train, Validation and Test")
    userResponse = str(input("Enter an option: "))
    if userResponse == "-1":
        mainQuestions()
    if userResponse == "1":
        T_V()
    if userResponse == "2":
        T_V_T()
    if userResponse == "-2":
        exit()


def Download():
    global data
    print()
    r = str(input("Enter yes to download: "))
    if r == "yes" or r == "y":
        data.to_csv("preprocessedData.csv", index=False)
    print()
    print("Downloaded!")
    print()
    userResponse = str(input("Enter an option: "))
    if userResponse == "-1":
        mainQuestions()
    if userResponse == "-2":
        exit()


file = sys.argv[1]
data = pd.read_csv(file)
data.columns = data.columns.str.replace(' ', '')

print()
print("Welcome to Data Processing CLI!")
print()
print("Press -1 to return to the previous options")
print("Press -2 to end the Program")
print("Note: The names of the columns in your dataset have been modified to exclude white spaces")
print()

mainQuestions()
