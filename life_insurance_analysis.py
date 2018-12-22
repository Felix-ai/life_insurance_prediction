# ========================================
# LIFE INSURANCE ANALYSIS
# ========================================


# Import Modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import os
from lifelines import KaplanMeierFitter
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# Define Functions
# Import Dataset
def load_data():
    PATH = Path("data/")
    data = pd.read_csv(PATH / "Obituaries_Dataset.csv",
                       parse_dates=["Announcement", "Death", "Burial"])

    # Count Missing Values
    ratio_missing = data.isnull().sum().sort_values(ascending=False) / len(data)

    # EDA
    print("-" * 70)
    print("Raw Dataset:")
    print(data.tail().transpose())
    print("-" * 70)

    print("Dataset Dimension:", data.shape)
    print("-" * 70)

    print("Descriptive Statistics:")
    print(data.describe(include="all").transpose())
    print("-" * 70)

    print("Missing Values Ratio:")
    print(ratio_missing)
    print("-" * 70)

    return data


# Retrieve Features and Targets
def features_target(data):
    df = data[pd.notnull(data["Fundraising"])]
    clms = ["Fundraising", "Name", "Announcement", "Death", "Burial",
            "Cost_Morgue", "Residence_Category", "Residence", "Residence_Name",
            "Corporate", "Repetition", "Corporate_Name", "Same_Morgue",
            "Occupation", "Hospital", "Distance_Death", "Age", "County_Death"]
    X = df.drop(clms, axis=1)
    y = df["Fundraising"]

    cd = df.groupby(df["Fundraising"]).count() / len(df)

    # EDA
    print("Data Types:")
    print(X.dtypes)
    print("-" * 70)

    print("Features Shape:", X.shape)
    print("Targets Shape:", y.shape)
    print("-" * 70)

    print("Class Distribution:")
    print(cd["Name"])
    print("-" * 70)

    return X, y


def enc_dec(X, y):
    cat = ["Burial_Day", "Burial_Week", "Gender", "Color", "County_Burial",
           "County_Morgue", "Cause_of_Death", "Married", "Spouse_Alive",
           "Spouse_gender", "Morgue"]
    ints = ["Word_Count", "Death_to_Announce", "Death_to_Burial",
            "Announce_to_Burial", "No_of_Relatives", "Distance_Morgue"]

    X_e = pd.DataFrame()
    for c in cat:
        lef = LabelEncoder()
        X_e[c] = lef.fit_transform(X[c].astype("str"))

    let = LabelEncoder()
    y_e = let.fit_transform(y.astype("str"))

    # Store Preprocessed Dataset
    os.makedirs("tmp", exist_ok=True)
    X_e.to_feather("tmp/features-raw")

    # EDA
    print("Target Classes:")
    print(let.classes_)
    print("-" * 70)

    return X_e, y_e, lef, let


# Split to Training, Validation and Testing Datasets
def split_val(X_o, y_o):
    X, y = SMOTE().fit_sample(X_o, y_o)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.01,
                                                        random_state=1)

    return X_train, X_test, y_train, y_test


# Train the Model
def model(X, X_test, y, y_test):
    # Fit Model
    clf = RandomForestClassifier(n_estimators=100, max_depth=7,
                                 bootstrap=False, n_jobs=-1, oob_score=False)

    clf.fit(X_train, y_train)
    train_a = clf.score(X_train, y_train)

    # EDA
    print("Training Accuracy: \t{:.4f}".format(train_a))
    print("-" * 70)

    # Test Results
    # Make Predictions
    y_pred = clf.predict(X_test)
    test_a = clf.score(X_test, y_test)
    creport = classification_report(y_test, y_pred)

    # Note that in binary classification,
    # recall of the positive class is also known as “sensitivity”;
    # recall of the negative class is “specificity”.
    cmatrix = confusion_matrix(y_test, y_pred)

    print("Test Accuracy: {:.4f}".format(test_a))
    print("-" * 70)

    print("Classification Report:")
    print(creport)
    print("-" * 70)

    print("Confusion Matrix:")
    print(cmatrix)
    print("-" * 70)


# Kaplan-meier Survival Curve
def KapMeir(df_raw, let):
    # set some plotting aesthetics
    sns.set(palette="colorblind", font_scale=1.35,
            rc={"figure.figsize": (12, 9), "axes.facecolor": ".92"})

    df = df_raw.loc[:, ["Age", "Fundraising"]].dropna(axis=0)
    eo = let.transform(df["Fundraising"])

    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["Age"], event_observed=eo)

    # plot the KM estimate
    kmf.plot()

    # Add title and y-axis label
    plt.title("Kaplan-Meier Estimate")
    plt.ylabel("Probability")

    plt.show()


# Run Script
if __name__ == "__main__":
    data_raw = load_data()
    features, label = features_target(data_raw)
    features_e, label_e, lef, let = enc_dec(features, label)
    X_train, X_test, y_train, y_test = split_val(features_e, label_e)
    model(X_train, X_test, y_train, y_test)
    KapMeir(data_raw, let)
