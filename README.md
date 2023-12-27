# Data preprocessing with CLI
This project contains the single python file needed to run a CSV data preprocessing programme in the command line interface.

## Quick Start
Download the main.py file and place it in a folder alongside your desired CSV file. Open the terminal and run the following command:
> <code> python3 main.py filename.csv </code>

## Preprocessing options 
After opening the file, the interface will provide certain preprocessing options. 
Remember: press -1 to return back to the previous options,
          press -2 to quite the program. Always remember to download the preprocessed data before quiting.

### Dataset Statistics
- Column statistics (enter column name)
  - Count of elements
  - Mean
  - Median
  - Mode
  - Skewness
  - Kurtosis
  - Standard Deviation
  - Inter-quartile Deviation
  - Range (minimum, maximum)

- Dataset Statistics
  - Visualise histograms (if dtype == numeric)
  - Column data types

### Null Value Handling
- View Null Values
  - Prints the columns and their corresponding null values

- Drop Null Values (WARNING: it'll either remove rows with null values OR drop entire columns with more than 75% null values)
  - Drop the null values
  - Prints the columns with corresponding null values

- Replace Null values (choose)
  - replace with mean
  - replace with mode
  - replace with median
    
### Encoding Categorical Data
- One-hot encoding
  - Creates columns for categorical data with  _column_feature_  as the column name
  - Prints all the columns names
    
### Feature Scaling
- Minimax scaler
  - Uses formula [(x - min(x))/(max(x) - min(x))]

- Standard scalre
  - Uses formula [(x - mean(x))/std(x)]
    
### Splitting the Data
- Train, Test (enter column names and ratio)
  - Splits the columns selected into train and test according to the ratio provided (test_split)
  - Downloads the data as train.csv and test.csv

- Train, validation and Test (enter column names, validation ratio and test ratio)
  - Splits the columns into first the train and temporary dataset (according to the first ratio)
  - Splits the temporary dataset into validation and test dataset (according to second ratio)
  - Downloads the data as train.csv, validation.csv and test.csv
    
### Download the data
- Downloads the preprocessed data as "preprocessedData.csv" .
