import pandas as pd
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Get the current date and time
now = datetime.datetime.now()

# Format the current date and time as a string
timestamp_str = now.strftime("%Y%m%d_%H%M%S")

# Create a PdfPages object with the current timestamp in the file name
pdf_pages = matplotlib.backends.backend_pdf.PdfPages(f'zadanie_3_{timestamp_str}.pdf')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# read dane_dla_stundetow.csv
def read_data_studends():
    with open('dane_dla_studentow.csv', 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large
        print(result)
    data = pd.read_csv('dane_dla_studentow.csv', encoding='windows-1250', sep=';')
    return data




students_data = read_data_studends()

def read_data_dodatkowe():
    data = pd.read_csv('dodatkowe_dane.csv', encoding='UTF-8', sep=',')
    return data

additional_data = read_data_dodatkowe()

students_data.columns = additional_data.columns

combined_data = students_data._append(additional_data, ignore_index=True)


# check data types and column structure
def check_data_structure(data):
    return data.info()


print(check_data_structure(combined_data))


# change values in columns to correct format
def change_values(data):
    repl_wykszt = {
        'WyĹĽsze': 'Wyższe',
        'Ĺšrednie': 'Średnie',
        'Tehnik': 'Technik'
    }
    repl_stan = {
        'InĹĽynier': 'Inżynier'
    }

    repl_kk = {
        '2138"': '2138'
    }
    data['Wykształcenie'] = data['Wykształcenie'].replace(repl_wykszt).astype(str)
    data['Stanowisko'] = data['Stanowisko'].replace(repl_stan).astype(str)
    data['Karty_kredytowe'] = data['Karty_kredytowe'].replace(repl_kk).astype(str)
    data['Karty_kredytowe'] = data['Karty_kredytowe'].astype(int)
    return data




combined_data = change_values(combined_data)

print(combined_data.head(10))

# check for duplicates (ID)
def check_duplicates(data):
    return data.duplicated(subset='ID').sum()


duplicates = check_duplicates(combined_data)
print(f"Number of duplicates: {duplicates}")

# Calculate the average age
average_age = combined_data['Wiek'].mean()
print(f"Average age: {average_age}")

# Get the sex distribution
sex_distribution = combined_data['Płeć'].value_counts()
print("Sex distribution:\n", sex_distribution)

# Find the most common education level
dominant_education = combined_data['Wykształcenie'].mode()[0]
print(f"Dominant education level: {dominant_education}")
# Plot the sex distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=sex_distribution.index, y=sex_distribution.values, palette="viridis")
plt.title('Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
pdf_pages.savefig(plt.gcf())
plt.close()

# Plot the age distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=combined_data, x="Wiek", bins=30, color="skyblue", kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
pdf_pages.savefig(plt.gcf())
plt.close()

# Plot the education level distribution
education_distribution = combined_data['Wykształcenie'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=education_distribution.index, y=education_distribution.values, palette="viridis")
plt.title('Education Level Distribution')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=90)
pdf_pages.savefig(plt.gcf())
plt.close()
# Debt (sum of Kredyty and Karty Kredytowe) per education level
debt_per_education = combined_data.groupby("Wykształcenie")[["Kredyty", "Karty_kredytowe"]].sum()
debt_per_education = debt_per_education.assign(total_debt = debt_per_education['Kredyty'] + debt_per_education['Karty_kredytowe'])
debt_per_education_sorted = debt_per_education.sort_values(by=['total_debt'], ascending=False)
print("Debt per education level:\n", debt_per_education_sorted)


# Analysis of years of experience (Lata_doświadczenia) per position (Stanowisko) and income (Roczny_dochód)
# Calculate the average years of experience per position
avg_exp_per_position = combined_data.groupby("Stanowisko")["Lata_doświadczenia"].mean()
print("Average years of experience per position:\n", avg_exp_per_position)

# Calculate the average income per position
avg_income_per_position = combined_data.groupby("Stanowisko")["Roczny_dochód"].mean()
print("Average income per position:\n", avg_income_per_position)

# Calculate the average income per years of experience
avg_income_per_exp = combined_data.groupby("Lata_doświadczenia")["Roczny_dochód"].mean()
print("Average income per years of experience:\n", avg_income_per_exp)

# Calculate the average years of experience per education level
avg_exp_per_education = combined_data.groupby("Wykształcenie")["Lata_doświadczenia"].mean()
print("Average years of experience per education level:\n", avg_exp_per_education)


# create prediction model for Roczny_dochód per Lata_doświadczenia
def prepare_data(df, feature_col, target_col, test_size=0.2, random_state=42):
    """
    Prepare the data by splitting into training and testing sets.
    """
    X = df[[feature_col]]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """
    Train the linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Squared Error and R-squared metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

def plot_results(X_test, y_test, y_pred):
    """
    Plot the actual vs predicted results.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.plot(X_test, y_pred, color='black', linewidth=2)
    plt.xlabel('Lata_doświadczenia')
    plt.ylabel('Roczny_dochód')
    plt.legend()
    plt.title('Linear Regression: Experience vs Income')
    pdf_pages.savefig(plt.gcf())
    plt.close()

X_train, X_test, y_train, y_test = prepare_data(combined_data, 'Lata_doświadczenia', 'Roczny_dochód')
model = train_model(X_train, y_train)
mse, r2, y_pred = evaluate_model(model, X_test, y_test)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


def check_correlations(df, columns):
    """
    Calculate and plot the correlation matrix for the selected columns.
    """
    # Calculate the correlation matrix
    # check if columns is not int
    for col in columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]
    correlation_matrix = df[columns].corr()

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    pdf_pages.savefig(plt.gcf())
    plt.close()

    return correlation_matrix

corr_data = combined_data.copy()
columns = ['Wiek', 'Lata_doświadczenia', 'Roczny_dochód', 'Płeć', 'Wykształcenie', 'Stanowisko']

correlation_matrix = check_correlations(corr_data, columns)

print(correlation_matrix)


def expenses_to_income(df):
    # Get all columns with prefix "Wydatki_"
    expenses_columns = [col for col in df.columns if col.startswith('Wydatki_')]

    # Calculate total expenses
    df['total_expenses'] = df[expenses_columns].sum(axis=1)

    # Calculate expenses percentage in income
    df['expenses_percentage'] = (df['total_expenses'] / df['Roczny_dochód']) * 100

    # Plot expenses percentage in income
    plt.figure(figsize=(10, 6))
    plt.bar(df['Stanowisko'], df['expenses_percentage'])
    plt.xlabel('Stanowisko')
    plt.ylabel('Expenses Percentage in Income (%)')
    plt.title('Expenses Percentage in Income per Individual')
    pdf_pages.savefig(plt.gcf())
    plt.close()

expenses_to_income(combined_data)

def savings_to_income(df):
    # Calculate savings percentage in income
    df['savings_percentage'] = (df['Oszczędności'] / df['Roczny_dochód']) * 100

    # Plot savings percentage in income
    plt.figure(figsize=(10, 6))
    plt.bar(df['Stanowisko'], df['savings_percentage'])
    plt.xlabel('Stanowisko')
    plt.ylabel('Savings Percentage in Income (%)')
    plt.title('Savings Percentage in Income per Individual')
    pdf_pages.savefig(plt.gcf())
    plt.close()

savings_to_income(combined_data)


def debt_to_income(df):
    # Calculate debt percentage in income
    # calculate total_debt
    df['total_debt'] = df['Kredyty'] + df['Karty_kredytowe']
    df['debt_percentage'] = round((df['total_debt'] / df['Roczny_dochód']) * 100, 2)
    # Plot debt percentage in income
    plt.figure(figsize=(10, 6))
    plt.bar(df['Stanowisko'], df['debt_percentage'])
    plt.xlabel('Stanowisko')
    plt.ylabel('Debt Percentage in Income (%)')
    plt.title('Debt Percentage in Income per Individual')
    pdf_pages.savefig(plt.gcf())
    plt.close()

debt_to_income(combined_data)


def individual_expenses_to_income(df, groupby_columns):
    # Get all columns with prefix "Wydatek_"
    expenses_columns = [col for col in df.columns if col.startswith('Wydatki_')]

    # Initialize an empty dictionary to store the results
    result_dict = {}

    for column in groupby_columns:
        # Group by column and calculate the sum of each 'Wydatek_' and 'Roczny_dochód'
        grouped_df = df.groupby(column)[expenses_columns + ['Roczny_dochód']].sum()

        for expense in expenses_columns:
            # Calculate expense percentage in income
            grouped_df[expense + '_percentage'] = round((grouped_df[expense] / grouped_df['Roczny_dochód']) * 100, 2)

        # Add the grouped DataFrame to the result dictionary
        result_dict[column] = grouped_df

    return result_dict

groupby_columns = ['Płeć', 'Wykształcenie', 'Stanowisko']
result_dict = individual_expenses_to_income(combined_data, groupby_columns)

# Print each DataFrame in the result dictionary
for column, df in result_dict.items():
    print(f"\nGrouped by {column}:\n")
    print(df)




plot_results(X_test, y_test, y_pred)

pdf_pages.close()
