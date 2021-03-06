# ========================================
# LIFE INSURANCE ANALYSIS
# ========================================


# Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path
from subprocess import call

from lifelines import KaplanMeierFitter
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import export_graphviz


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
    # Categorical Variables
    cat = ["Burial_Day", "Burial_Week", "Gender", "Color", "County_Burial",
           "County_Morgue", "Cause_of_Death", "Married", "Spouse_Alive",
           "Spouse_gender", "Morgue"]

    # Continuous Variables
    ints = ["Word_Count", "Death_to_Announce", "Death_to_Burial",
            "Announce_to_Burial", "No_of_Relatives", "Distance_Morgue"]

    # Expressions to omit in Continuous Variables
    om = ["#REF!", "o", "#VALUE!", "9/29/1782", "10/15/1782", "10/20/1782",
          "7/30/1782", "11/19/1782", "5/25/2017", "5/27/2017", "5/26/2017",
          "2/24/2017", "11/16/1782", "2/18/2017", "11/21/1782", "2/17/2017",
          "11/15/1782", "2/23/2017", "2/25/2017", "2/27/2017", "10/22/1782",
          "3/17/2017", "3/20/2017", "3/18/2017", "10/18/1782", "3/15/2017",
          "3/14/2017", "10/29/1782", "3/31/2017", "3/29/2017", "3/30/2017",
          "10/14/1782", "3/21/2017", "3/25/2017", "3/22/2017", "11/27/1782",
          "11/26/1782"]

    # Initialize an Empty Dataframe
    X_e = pd.DataFrame()

    # Encode Categorical Variables
    for c in cat:
        lef = LabelEncoder()
        X_e[c] = lef.fit_transform(X[c].astype("str"))

    # Encode Float Variables Correctly
    for i in ints:
        X_e[i] = X[i].str.replace(",", "").replace(om, np.nan).astype(float)

    # Replace Missing Values with Variable Median
    X_e.fillna(X_e.mean(), inplace=True)

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
    X, y = SMOTE(random_state=11).fit_sample(X_o, y_o)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.01,
                                                        random_state=6)

    return X_train, X_test, y_train, y_test


# Train the Model
def model(X_train, X_test, y_train, y_test):
    # Fit Model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5,
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

    # Plot Confusion Matrix
    labels = ["No", "Yes"]
    ax = plt.subplot()
    sns.heatmap(cmatrix, annot=True, ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.show()

    return clf


def est_plot(clf, X_e, let):
    estimator = clf.estimators_[26]

    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot',
                    feature_names=X_e.columns.values,
                    class_names=let.classes_,
                    rounded=True, proportion=False,
                    precision=2, filled=True, max_depth=3)

    # Convert to png using system command (requires Graphviz)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'],
          shell=True)


# Kaplan-meier Survival Curve
def KapMeir(df_raw, let):
    plt.clf()

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
    clf = model(X_train, X_test, y_train, y_test)
    est_plot(clf, features_e, let)
    KapMeir(data_raw, let)
