import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

st.title("Intro + Background: MLB Scouting Methodologies and Performance Prediction")

st.write("""
MLB scouting departments often follow different methodologies around scouting. Whether to draft players out of high school or out of college is an important scouting decision. 
We want to predict the performance of a player based on NCAA statistics on how they’ll perform in the MLB. A plethora of data is available surrounding baseball performance. 
Hitting metrics are often focused on for contribution to team performance, but there are also defensive statistics available, and pitchers can be compared. 
It’s important to realize that a team’s number of outs available. This is a scarce resource, so whichever action a player can do to reduce the depletion of this resource is good for the team. 
A team’s goal should be to win games, and the best way to do this is to score runs. So the more runs a team can score, the more games they should win. [5].

Baseball is a sport rich with data, and because every team plays 162 games in a season, variance is lower with the larger sample size. Some popular hitting metrics include batting average 
(player hits / number of at bats), RBI (runs batted in), and home runs. OPS, “on-base plus slugging” is correlated with the number of runs scored, and is very popular in sabermetrics. 
It adds the on-base percentage and slugging percentage (total bases reached / at bats). These two metrics individually are also helpful to look at. Popular pitching metrics include wins 
(pitcher when a team takes the lead and doesn’t lose it, with some exceptions), ERA (earned run average), and strikeouts. [3, 4]
""")


st.title("Methods")

st.title("Preprocessing Methods")

st.markdown(
    """
### Encode Categorical Data:
To encode categorical data numerically for valid input for models we will use a One-Hot Encoder from `sklearn.preprocessing.OneHotEncoder`.

### Normalize Data:
To ensure features are all on the same scale we will use scaling from `sklearn.preprocessing.StandardScaler`.

### Select Relevant Features:
To identify and select the most relevant features to our classification from the data-set we will utilize Principal Component Analysis from `sklearn.decomposition.PCA`.

### Address Data Imbalance:
Due to the fact there are many more players in the NFL than College Football, we will have to rebalance our data. A recent study from researchers analyzing methods for rebalancing datasets found Synthetic Minority Oversampling Technique or SMOTE to be the most effective [1]. For this we can use `imblearn.over_sampling.SMOTE`.
"""
)

st.title("ML Algorithms/Models for Baseball Performance Prediction")

# Random Forest
st.header("Random Forest (Supervised)")
st.write("""
To predict player performance metrics in baseball such as batting average using NCAA data. Random forest is beneficial for data with a large number of features.
We can use the scikit-learn library with the function `RandomForestClassifier`.
""")

# Support Vector Machines (SVM)
st.header("Support Vector Machines (SVM) (Supervised)")
st.write("""
To predict categorical outcomes such as whether a collegiate player turns pro. SVM can help to predict binary outcomes.
We can use the scikit-learn library with the function `SVC`.
""")

# K-Means
st.header("K-Means (Unsupervised)")
st.write("""
To find players with similar characteristics and statistics. K-Means can help to cluster similar player profiles.
We can use the scikit-learn library with the function `KMeans`.
""")

# Gaussian Mixture Model (GMM)
st.header("Gaussian Mixture Model (GMM) (Unsupervised)")
st.write("""
To find players with similar characteristics and statistics. GMM can help to show clusters with varying shapes, densities, and overlaps which provide more flexibility for our data.
We can use the scikit-learn library with the function `GaussianMixture`.
""")

st.header("Potential Results + Discussion")

st.markdown(
    """
To test quantitatively, we'll use accuracy, precision, and F1 score to evaluate our ML model. Accuracy measures overall performance, precision measures true positive predictions, and the F1 score will balance precision and recall for us. This aligns with algorithms like Random Forest and Logistic Regression, which are suitable for binary classification. High precision and F1 scores would indicate that our model predicts MLB success reliably while addressing imbalances. [2] Project goals that we have are improving prediction accuracy by selecting relevant features and normalizing data, with expected results of balanced performance across all metrics. We would also like to create an end-to-end model which to somewhat accuracy predicts college/high-school to MLB success.
"""
)
st.write("[Click here to view the Gantt Chart](https://docs.google.com/spreadsheets/d/1u2V7eAzZfUiB1OnysyVkAIUWN8Op6MRT/edit?usp=sharing&ouid=112263630298576499385&rtpof=true&sd=true)")

tasks = {
    'Task': [
        'Project Proposal', 'Introduction & Background', 'Problem Definition', 'Methods', 
        'Potential Results & Discussion', 'Video Recording', 'GitHub Page',
        'Data Sourcing and Cleaning (Model 1)', 'Model Selection (Model 1)', 'Data Pre-Processing (Model 1)',
        'Model Coding (Model 1)', 'Results Evaluation and Analysis (Model 1)', 'Midterm Report',
        'Data Sourcing and Cleaning (Model 2)', 'Model Selection (Model 2)', 'Data Pre-Processing (Model 2)',
        'Model Coding (Model 2)', 'Results Evaluation and Analysis (Model 2)', 'Model Comparison', 
        'Presentation', 'Recording', 'Final Report'
    ],
    'Person': [
        'Josh', 'Josh', 'Josh', 'Anthony & Steven', 'Sai', 'Arnav', 'Sai', 
        'Steven', 'Josh', 'Sai', 'Anthony', 'Arnav', 'All', 
        'Steven', 'Josh', 'Sai', 'Anthony & Arnav', 'Arnav', 'Anthony', 
        'All', 'All', 'All'
    ],
    'Start Date': [
        '2024-09-27', '2024-09-27', '2024-09-27', '2024-09-27', '2024-09-27', '2024-09-27', '2024-09-27',
        '2024-10-07', '2024-10-15', '2024-10-18', '2024-10-25', '2024-11-08', '2024-11-08',
        '2024-10-18', '2024-10-22', '2024-10-25', '2024-10-25', '2024-11-19', '2024-11-29', 
        '2024-11-29', '2024-12-06', '2024-11-29'
    ],
    'Finish Date': [
        '2024-10-04', '2024-10-04', '2024-10-04', '2024-10-04', '2024-10-04', '2024-10-04', '2024-10-04',
        '2024-10-15', '2024-10-18', '2024-10-25', '2024-11-08', '2024-11-16', '2024-11-16',
        '2024-10-22', '2024-10-25', '2024-10-29', '2024-11-19', '2024-11-24', '2024-12-07', 
        '2024-12-06', '2024-12-07', '2024-12-07'
    ]
}

# Create DataFrame
df = pd.DataFrame(tasks)

# Display the DataFrame in Streamlit
st.title("Gantt Chart for Baseball Project (Table Format)")
st.write(df)
st.header("Contribution Table")

data = {
    'Person': ['Josh Forden', 'Anthony Pastrana', 'Saisaketh Koppu', 'Steven Hao', 'Arnav Chintawar'],
    'Tasks Completed': ['Intro and Background and Baseball Ideation', 'Preprocessing Methods', 'Streamlit + Github Creation and Results + Discussion', 'ML Algorithms/Models', 'Video and Slides']
}

df = pd.DataFrame(data)

# Display the table in Streamlit
st.title("Project Tasks Completed")
st.write(df)

st.header("References")

st.markdown(
    """
1. Y. F. Roumani, "Sports analytics in the NFL: classifying the winner of the Superbowl," Annals of Operations Research, vol. 325, no. 1, pp. 715–730, 2023, doi: 10.1007/s10479-022-05063-x.
2. N. Sharma, “Understanding and Applying F1 Score: A Deep Dive with Hands-On Coding,” Arize AI, Jun. 06, 2023. https://arize.com/blog-course/f1-score/
3. J. Zimmerman, “A new way to look at College Players’ Stats,” The Hardball Times,  https://tht.fangraphs.com/a-new-way-to-look-at-college-players-stats/ (accessed Oct. 4, 2024). 
4. “Baseball statistics, ” Baseball Reference, Feb. 23, 2024. https://www.baseball-reference.com/bullpen/Talk:Baseball_statistics
5. “Pythagorean Winning Percentage,” MLB Advanced Media, 2024. https://www.mlb.com/glossary/advanced-stats/pythagorean-winning-percentage
"""
)
