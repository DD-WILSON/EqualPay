# EqualPay
A data analysis project exploring wage gaps and pay equity trends, using SQL and data visualization tools

## 📊 Project Overview
This project investigates pay equity and wage gap trends across various industries and demographics. It uses publicly available datasets, SQL queries for data manipulation, and visualization tools to highlight key findings.

## 📝 Objectives
- Analyze wage gaps by gender and industry.
- Identify trends in pay equity over time.
- Build visualizations to communicate insights effectively.

## 📂 Project Structure
📁 data/
📁 notebooks/
📁 queries/
📁 visualizations/
📄 README.md

markdown
Copy
Edit

## 🛠️ Tools & Technologies
- SQL
- Python (Pandas, Matplotlib, Seaborn)
- Excel
- Tableau / Power BI (if applicable)

## 📈 Example Insights
- "Women in the tech industry earn X% less than their male counterparts."
- "The pay gap has narrowed by Y% in the healthcare sector since 2010."

## 💡 Future Work
- Incorporate intersectional analysis (e.g., by race/ethnicity).
- Predict future pay gap trends using regression models.

## 📃 License
This project is licensed under the MIT License.




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
# 2. Define and import the CSV Data file
import pandas as pd

# Specify the full path to the CSV file
df = pd.read_csv(r"C:\Users\342338\Documents\Regression Data Test Model 2.csv")
# 3. Encoding and Data cleaning
OHE = 'One Hot Encoding'
DF = 'Data Frame'
from sklearn.preprocessing import OneHotEncoder
# Setting parameters to ignore errors and place the outputs into the dataframe
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# One Hot Encoding DG Ship
ohetransform = ohe.fit_transform(df[['DG Ship']])
# Concating the newly OHE DG Ship with the current dataframe
df = pd.concat([df, ohetransform], axis=1). drop(columns = ['DG Ship'])
# Dropping any null values from DG Ship
df.drop(columns=['DG Ship_nan'], errors='ignore', inplace=True)
# Numerising 'Gender', Female as 1, Male as 0
df['Gender'] =df['Gender'].map({'Female': 0, 'Male': 1})
#Defining Premium and Non Premium Reference Points, 1 as Premium, 0 as non Premium
mapping = {
    'Corporate (Other)': 1,
    'Central Banking': 0,
    'Legal': 1,
    'External Premium Technology': 1,
    'Corporate': 0,
    'Maintenance': 0,
    'Research': 1,
    'Technology': 0,
    'Actuaries': 1,
    'External Premium Central Banking': 1,
    'Security': 0,
    'Industrial Placement': 0,
    'SENIOR ADVISOR': 1,
    'SPECIAL ADVISOR': 1,
    'DEPUTY GOVERNOR': 1,
    'GOVERNOR': 1
}

# Use the .map() function to replace old reference point values with new numerised values
df['Reference Point'] = df['Reference Point'].map(mapping)
# Mapping Scale to numerised values
position_mapping = {
    'Scale A': 10,
    'Scale B': 10,
    'Scale C': 9,
    'Scale D': 8,
    'Scale E': 7,
    'Scale F': 6,
    'Scale G': 5,
    'Scale H': 4,
    'Scale I': 3,
    'Scale J': 2,
    'Scale K': 1,
}

# Use the .map() function to replace old Grade values with new numerised values
df['Level or Grade'] = df['Level or Grade'].map(position_mapping)

# Replacing Manager Responsibility with numerised values, 0 for No responsibilty and 1 for being a manager
df['Manager Responsibilities'] =df['Manager Responsibilities'].map({'N': 0, 'Y': 1})
# Create a numersied 'New Joiner Column' using .map() 
df['New Joiner Column'] = df['Changed role in last 12 months?'].map({'New Joiner': 1, 'Not promoted': 0, 'Promoted': 0})

# Create a numerised 'Promoted Column' using .map()
df['Promoted Column'] = df['Changed role in last 12 months?'].map({'Promoted': 1, 'Not promoted': 0, 'New Joiner': 0})

# Data Cleaning replacing Too Soon to Rate inconsistencies
df['Performance Rating - Last Annual Review'] = df['Performance Rating - Last Annual Review'].replace("Too soon to rate (TSTR)", "Too Soon To Rate (TSTR)")
# Replacing any null values in Performance rating column with 0's, the fillna command replaces the original df with the newly formed values
df['Performance Rating - Last Annual Review'] = df['Performance Rating - Last Annual Review'].fillna(0)
# importing numpy package
import numpy as np

# Sample array
arr = np.array(['Too Soon To Rate (TSTR)', 'Succeeding', 'Excelling', 'succeeding',
                'Developing', 'Underperforming', 0, '0'], dtype=object)

# Replace '0' with 0
arr = np.where(arr == '0', 0, arr)
# Formatting and replacing text 0 with number 0
df['Performance Rating - Last Annual Review'] = df['Performance Rating - Last Annual Review'].replace('0',0)
# Replacing text errors within performance rating
df['Performance Rating - Last Annual Review'] = df['Performance Rating - Last Annual Review'].replace('succeeding','Succeeding')
#Numerising performace rating
rating_mapping = {
    0: 0,  # Keep 0 as 0
    'Too Soon To Rate (TSTR)': 1,
    'Underperforming': 2,
    'Developing': 3,
    'Succeeding': 4,
    'Excelling': 5
}

# Map the performance ratings to numeric values in the df
df['Performance Rating - Last Annual Review'] = df['Performance Rating - Last Annual Review'].map(rating_mapping)
#OHE Ethncity
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
ohetransform1 = ohe.fit_transform(df[['Ethnicity']])
# Concat the newly transformed Ethnicity column with the df and dropping the old column.
df = pd.concat([df, ohetransform1], axis=1).drop(columns=['Ethnicity'])

# Combining the newly transformed Not Ethnic Minority and Ethnic Minority column into a single column
df['Ethnicity_Binary'] = ohetransform1['Ethnicity_Ethnic Minority'] + ohetransform1['Ethnicity_Not Ethnic Minority']

# Drop old columns from the df
df.drop(columns=['Ethnicity_Ethnic Minority', 'Ethnicity_Not Ethnic Minority', 'Ethnicity_Not declared', 'Ethnicity_Prefer not to say'], errors='ignore', inplace=True)
 #One-Hot Encode Disability Status
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
ohe_transform = ohe.fit_transform(df[['Disability Status']])

# Add OHE columns to df and drop the original column
df = pd.concat([df, ohe_transform], axis=1).drop(columns=['Disability Status'])

# Define Disability_Binary: 1 = No Disability, 0 = Has Disability
df['Disability_Binary'] = df['Disability Status_No']

# Drop unnecessary OHE columns
df.drop(columns=['Disability Status_Yes', 'Disability Status_No',
                 'Disability Status_Unknown', 'Disability Status_Prefer not to say'],
        errors='ignore', inplace=True)
# Numersing Part time as 0 and Full time as 1
df['Working time '] =df['Working time '].map({' PT': 0, ' FT': 1})
# This drops any columns that have all values as null and makes these changes to the original df
df.dropna(axis=1, how='all', inplace=True)
# Allow Pandas to display an unlimited quantity of columns
pd.set_option('display.max_columns', None)
#Cleans any trailing or proceeding spaces in all columns of the df
df.columns = df.columns.str.strip()
# Dropping defined columns and replaces makes sure these changes are made to the original df
df.drop(columns=['Salary range min', 'Salary range max', 'NP Allowance'], errors='ignore', inplace=True)
# Dropping defined columns and replaces makes sure these changes are made to the original df
df.drop(columns=['Unnamed: 32', 'Salary Sacrifice?', 'Salary Sacrifice, Working', 'Pens Flex Up Override', 'Pens Flex Up', 'SUPP Pension', 'Supp Pension Override', 'Childcare Vouchers', 'Additional Leave Deduction', 'Cycle to work', 'Cycle to Work Override', 'Shift pay', 'Acting Up', 'Acting Up Override', 'Benefits Allowance', 'Benefits Allowance Override', 'Car Allowance', 'Deb Attend', 'DebdenMKTS', 'INCWDAY Retroactive', 'INCWEND Retroactive', 'NP Allowance', 'Pens Flex Down', 'Pens Flex Down Override', 'PLO Salary', 'Rent All', 'Serv Award', 'Shift Allowance', 'Shift Supp', 'Skills Premium', 'StCFnd Salary', 'Period of time Annual Bonus is for', 'Eligible for Bonus 1/Bonus ASR?', 'Date of Annual Bonus.1', 'Amount paid - Annual Bonus.1', 'Eligible for Bonus 2/Governors Award?', 'Amount paid - Bonus 2', 'Eligible for Bonus 3/Skills Premium?', 'Amount paid - Bonus 3', 'Eligible for Bonus 4/Ex Gratia Payment?', 'Amount paid - Bonus 4'],errors='ignore',inplace=True)
# Dropping defined columns and replaces makes sure these changes are made to the original df
df.drop(columns=['Most Recent Start Date', 'Returner Status', 'Pay frequency', ], errors='ignore', inplace=True)
# Dropping defined columns and replaces makes sure these changes are made to the original df
df.drop(columns=['Working hours', 'Week worked', 'Closest Connection', 'Job Title'], errors='ignore', inplace=True)
# Converting time in role values to Integers or Float values
df['Time in Role'] = pd.to_numeric(df['Time in Role'], errors='coerce')
df.dropna(subset=['Time in Role'], inplace=True)

# Rename the column 'Time in Role' to 'Time in Scale'
df.rename(columns={'Time in Role': 'Time in Scale'}, inplace=True)
# loc is used to locate values in a column using a condition and then replacing all Time in role values greater than 9 as 9
df.loc[df['Time in Scale'] > 9, 'Time in Scale'] = 9
# Dropping defined columns and replaces makes sure these changes are made to the original df
df.drop(columns=['Last pay review increase','Job Function'], errors='ignore', inplace=True)
#Defining Age as 2024 minus year of birth
df['Age'] = 2024 - df['Year of Birth']
# Ensuring no null values in Manager Responsibility column
df['Manager Responsibilities'] = df['Manager Responsibilities'].fillna('0')
# Changing the values in Manager Responsiiblites to INT values
df['Manager Responsibilities'] =df['Manager Responsibilities'].astype(int)
#Grouping different leave types into either Family leave, Sick Leave or Other Leave
family_leave = ['Maternity Leave', 'Paternity Leave Birth', 'Shared Parental Leave Birth',
                'Paternity Leave Adoption', 'Unpaid Parental Leave', 'Unpaid Assignment - Secondment']

sick_leave = ['Sickness Leave', 'Paid Leave']  # Assuming 'Paid Leave' is used for sickness

other_leave = ['Unpaid Leave 4 Weeks or Less', 'Unpaid Leave Greater than 4 Weeks']

# Create new columns for each category, the lamda function checked if the value appears in the defined list
df['Family_Leave'] = df['Leave Type'].apply(lambda x: 1 if x in family_leave else 0)
df['Sick_Leave'] = df['Leave Type'].apply(lambda x: 1 if x in sick_leave else 0)
df['Other_Leave'] = df['Leave Type'].apply(lambda x: 1 if x in other_leave else 0)
# Dropping defined columns and replaces makes sure these changes are made to the original df
df.drop(columns=['Leave Type','Year of Birth','Tenure','Unnamed: 33'], errors='ignore', inplace=True)
# Removes and null values from Level or Grade column
df = df.dropna(subset=['Level or Grade'])
# List of employee IDs to delete
ids_to_delete = [932350, 332615, 329265, 324758, 325172, 325725, 327010, 322866, 928168, 306795, 268193]

# Delete rows where 'Employee ID' is in the list of IDs to delete
df = df[~df['Employee ID'].isin(ids_to_delete)]
# 4. Check Multicolinearity
# Defining columns to check multicolinearity, i.e the correlation between columns to check if two columns are alike. All leaves removed as P values were too high
columns = ['Structural Adjustements',
 'Time in Scale',
 'Reference Point',
 'Level or Grade',
 'Manager Responsibilities',
 'Performance Rating - Last Annual Review',
 'Gender',
 'New Joiner Column',
 'Promoted Column',
 'Ethnicity_Binary',
 'Age',
]

# Calculate the correlation matrix for selected columns
corr_matrix = df[columns].corr()

# Print the entire correlation matrix to check correlation values
print(corr_matrix)
# Calculate the correlation matrix
correlation_matrix = df[columns].corr()

# Filter the matrix: keep only correlations with magnitude > 0.7
filtered_matrix_cleaned = correlation_matrix[correlation_matrix.abs() > 0.7].fillna("")

# Display the correlation matrix
print(filtered_matrix_cleaned)
# Printing all data types of the columns within the df in order to check if values are INT or Floats
df.info()
# 5. Split data into training and test. 

## By maintaining a clear distinction between training and test data, we can make sure that the model is not just good at memorizing the training data but capable of making accurate predictions on previously unseen data, which is the goal of machine learning.
# Split data into training and testing sets

# log transform y variables
#We log transform our data here as income has a skewed distribution, by transforming the data the model can more accurately predict income
import numpy as np

y_log = np.log(df['Contractual Annual Salary'])
y_sqred = np.sqrt(df['Contractual Annual Salary'])

X = df[columns]
y = df['Contractual Annual Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Code to illustrate the original distribution, a squared transformation, and a logarithmic transformation.
f, (ax0, ax1, ax2) = plt.subplots(1, 3)

ax0.hist(y, bins=100, density=True)
ax0.set_ylabel("Probability")
ax0.set_xlabel("Target")
ax0.set_title("Target distribution")

ax1.hist(y_sqred, bins=100, density=True)
ax1.set_ylabel("Probability")
ax1.set_xlabel("Target")
ax1.set_title("Transformed target distribution")

ax2.hist(y_log, bins=100, density=True)
ax2.set_ylabel("Probability")
ax2.set_xlabel("Target")
ax2.set_title("Transformed target distribution")

f.suptitle("Salary Distribution", y=1.05)
plt.tight_layout()
# Importing Sklearn package
from sklearn.preprocessing import StandardScaler
# Selecting the columns to standardize, this ensures all numbers have comparable scales
columns_to_standardize = ['Structural Adjustements', 'Time in Scale', 'Age']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the training data for the selected columns, fitting calculates the mean and transform standardises the data
X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])
X_test[columns_to_standardize] = scaler.transform(X_test[columns_to_standardize])
# Check for NaN values in the entire DF
nan_counts = df.isna().sum()

# Check for infinite values (both positive and negative infinity) in the entire DataFrame
inf_counts_pos = (df == float('inf')).sum()  # Positive infinity
inf_counts_neg = (df == float('-inf')).sum()  # Negative infinity

# Display results
print("Missing (NaN) values per column:")
print(nan_counts)

print("\nPositive infinity values per column:")
print(inf_counts_pos)

print("\nNegative infinity values per column:")
print(inf_counts_neg)
# Fit the model using statsmodels to get p-values, R-squared, etc.
# Add a constant to the model (intercept)
X_train_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_const).fit()

# Print model summary
print(model.summary())
# Step 5: Validate the model on the test dataset
# Add a constant to test data for consistency
X_test_const = sm.add_constant(X_test)
y_pred = model.predict(X_test_const)

# Model performance on test set
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", metrics.root_mean_squared_error(y_test, y_pred))
print("R-squared (test data):", metrics.r2_score(y_test, y_pred))

# Extract coefficients from the full-sample regression model
beta_existing = model.params  # Coefficients from full dataset

# Define independent variables (excluding Gender)
Newcolumns = [
    'Structural Adjustements', 'Time in Scale', 'Reference Point', 'Level or Grade',
    'Manager Responsibilities', 'Performance Rating - Last Annual Review',
    'New Joiner Column', 'Promoted Column', 'Age'
]

# Split data by Gender
male_data = df[df['Gender'] == 1]
female_data = df[df['Gender'] == 0]

# Extract headcounts
headcount_male = len(male_data)
headcount_female = len(female_data)

# Ensure there are both males and females in the dataset
if male_data.empty or female_data.empty:
    print("Insufficient data for gender analysis.")
else:
    # Extract X and y for Gender analysis
    X_male = male_data[Newcolumns]
    X_female = female_data[Newcolumns]
    y_male = male_data['Contractual Annual Salary']
    y_female = female_data['Contractual Annual Salary']

    # Add constant term (intercept)
    X_male_const = sm.add_constant(X_male)
    X_female_const = sm.add_constant(X_female)

    # Fit separate regression models for men and women
    model_male = sm.OLS(y_male, X_male_const).fit()
    model_female = sm.OLS(y_female, X_female_const).fit()

    # Extract separate male and female coefficient estimates
    beta_male = model_male.params
    beta_female = model_female.params

    # Compute means of independent variables
    mean_X_male = X_male_const.mean()
    mean_X_female = X_female_const.mean()

    # Calculate total wage gap
    total_gap_gender = y_male.mean() - y_female.mean()
    total_gap_gender_percent = (total_gap_gender / y_female.mean()) * 100  # % of female average salary

    # Oaxaca decomposition (Gender)
    explained_gender = (mean_X_male - mean_X_female) * beta_female
    unexplained_gender = mean_X_male * (beta_male - beta_female)

    explained_gender_sum = explained_gender.sum()
    unexplained_gender_sum = unexplained_gender.sum()
    explained_gender_percent = (explained_gender_sum / total_gap_gender) * 100
    unexplained_gender_percent = (unexplained_gender_sum / total_gap_gender) * 100
    adjusted_gap_gender_percent = total_gap_gender_percent * (1 - (explained_gender_percent / 100))

    # Breakdown of explained component
    explained_gender_breakdown = (explained_gender / explained_gender_sum) * 100

    # Print overall results
    print(f"Total Gender Pay Gap: {total_gap_gender_percent:.2f}%")
    print(f"Explained Gender Component: {explained_gender_percent:.2f}%")
    print(f"Unexplained Gender Component: {unexplained_gender_percent:.2f}%")
    print(f"Adjusted Gender Pay Gap: {adjusted_gap_gender_percent:.2f}%")
    print(f"Headcount (Male): {headcount_male}")
    print(f"Headcount (Female): {headcount_female}")
    print("Contribution of Each Factor to Explained Gender Gap:")
    print(explained_gender_breakdown.sort_values(ascending=False))

# Extract coefficients from the full-sample regression model
beta_existing = model.params  # Coefficients from full dataset

# Define independent variables (excluding Gender)
Newcolumns = [
   'Structural Adjustements', 'Time in Scale', 'Reference Point', 'Level or Grade',
   'Manager Responsibilities', 'Performance Rating - Last Annual Review',
   'Working time', 'New Joiner Column', 'Promoted Column', 'Age'
]

# Define the mapping of levels to scales
level_to_scale = {
   10: "Scale A/B",
   9: "Scale C",
   8: "Scale D",
   7: "Scale E",
   6: "Scale F",
   5: "Scale G",
   4: "Scale H",
   3: "Scale I",
   2: "Scale J",
   1: "Scale K"
}

# Dictionary to store results
results = {}

# Loop through each level (Scale A to Scale K, represented as 10 down to 1)
for level in range(10, 0, -1):
   df_level = df[df['Level or Grade'] == level]

   # Split data by Gender
   male_data = df_level[df_level['Gender'] == 1]
   female_data = df_level[df_level['Gender'] == 0]

   # Extract headcounts
   headcount_male = len(male_data)
   headcount_female = len(female_data)

   # Ensure there are both males and females in the level, otherwise skip
   if male_data.empty or female_data.empty:
       continue

   # Extract X and y for Gender analysis
   X_male = male_data[Newcolumns]
   X_female = female_data[Newcolumns]
   y_male = male_data['Contractual Annual Salary']
   y_female = female_data['Contractual Annual Salary']

   # Add constant term (intercept)
   X_male_const = sm.add_constant(X_male)
   X_female_const = sm.add_constant(X_female)

   # Fit separate regression models for men and women
   model_male = sm.OLS(y_male, X_male_const).fit()
   model_female = sm.OLS(y_female, X_female_const).fit()

   # Extract R-squared values
   r2_male = model_male.rsquared
   r2_female = model_female.rsquared

   # Extract separate male and female coefficient estimates
   beta_male = model_male.params
   beta_female = model_female.params

   # Compute means of independent variables
   mean_X_male = X_male_const.mean()
   mean_X_female = X_female_const.mean()

   # Calculate total wage gap
   total_gap_gender = y_male.mean() - y_female.mean()
   total_gap_gender_percent = (total_gap_gender / y_female.mean()) * 100  # % of female average salary

   # Oaxaca decomposition (Gender)
   explained_gender = (mean_X_male - mean_X_female) * beta_female
   unexplained_gender = mean_X_male * (beta_male - beta_female)

   explained_gender_sum = explained_gender.sum()
   unexplained_gender_sum = unexplained_gender.sum()
   explained_gender_percent = (explained_gender_sum / total_gap_gender) * 100
   unexplained_gender_percent = (unexplained_gender_sum / total_gap_gender) * 100
   adjusted_gap_gender_percent = total_gap_gender_percent * (1 - (explained_gender_percent / 100))

   # Breakdown of explained component
   explained_gender_breakdown = (explained_gender / explained_gender_sum) * 100

   # Store results
   results[level_to_scale[level]] = {
       "Gender Pay Gap (%)": total_gap_gender_percent,
       "Gender Explained (%)": explained_gender_percent,
       "Gender Unexplained (%)": unexplained_gender_percent,
       "Gender Adjusted Pay Gap (%)": adjusted_gap_gender_percent,
       "Headcount (Male)": headcount_male,
       "Headcount (Female)": headcount_female,
       "Gender Explained Breakdown": explained_gender_breakdown.sort_values(ascending=False),
       "R-squared Male": r2_male,
       "R-squared Female": r2_female
   }

   # Print results for this level
   print(f"\n{level_to_scale[level]}:")
   print(f"Total Gender Pay Gap: {total_gap_gender_percent:.2f}%")
   print(f"Explained Gender Component: {explained_gender_percent:.2f}%")
   print(f"Unexplained Gender Component: {unexplained_gender_percent:.2f}%")
   print(f"Adjusted Gender Pay Gap: {adjusted_gap_gender_percent:.2f}%")
   print(f"Headcount (Male): {headcount_male}")
   print(f"Headcount (Female): {headcount_female}")
   print(f"R-squared (Male model): {r2_male:.3f}")
   print(f"R-squared (Female model): {r2_female:.3f}")
   print("Contribution of Each Factor to Explained Gender Gap:")
   print(explained_gender_breakdown.sort_values(ascending=False))
   print("=" * 50)

