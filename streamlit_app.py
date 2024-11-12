import streamlit as st
import pandas as pd
import math
from pathlib import Path
from PIL import Image

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
A team’s goal should be to win games, and the best way to do this is to score runs. So the more runs a team can score, the more games they should win. [3].

Baseball is a sport rich with data, and because every team plays 162 games in a season, variance is lower with the larger sample size. Some popular hitting metrics include batting average 
(player hits / number of at bats), RBI (runs batted in), and home runs. OPS, “on-base plus slugging” is correlated with the number of runs scored, and is very popular in sabermetrics. 
It adds the on-base percentage and slugging percentage (total bases reached / at bats). These two metrics individually are also helpful to look at. Popular pitching metrics include wins 
(pitcher when a team takes the lead and doesn’t lose it, with some exceptions), ERA (earned run average), and strikeouts. [1, 2]
""")

st.title("Problem Definition")

st.write(""" 
Our project seeks to predict MLB player success based on NCAA performance metrics,
 focusing on both hitting and pitching statistics that are central to team performance. 
 Key metrics such as OPS, ERA, and strikeouts offer insights into a player's contribution to run scoring and game-winning potential. 
 We're passionate about this to see if we can figure out patterns and correlations which can increase potential for future wins.
""")

st.header("Methods")
st.subheader("Preprocessing")
st.write("""
Before training our model, we standardized our data using `sklearn.preprocessing.StandardScaler`. Standardizing data is useful in this case 
because we have features on different scales, such as batting average and number of runs. Ensuring all features are on the same scale 
prevents features with larger scales from disproportionately influencing the model.
""")

st.subheader("Model")
st.write("""
For our model, we selected a **Random Forest Regressor** from `sklearn.ensemble.RandomForestRegressor`. This model offers several advantages 
for our use case:
- **Lower risk of overfitting**: By selecting a random subset of features for each decision tree, Random Forests have a reduced chance of overfitting, especially with a sufficient number of estimators.
- **High-dimensional data handling**: Random Forests are well-suited to manage datasets with many features, which aligns with our data.
- **Feature importance evaluation**: Random Forests make it easy to assess the contribution of each feature, helping us determine which statistics are most predictive of a player's professional batting average.
""")

st.header("Results")

st.write("""
After training our **Random Forest Regressor** with 100 estimators, we achieved the following results:
- **Mean Squared Error (MSE):** 0.00036267402500000077
- **R-squared (R²):** 0.974410089089329
""")

image1 = Image.open("1111.png")
image2 = Image.open("2222.png")
st.image(image1, use_container_width=True)
st.image(image2, use_container_width=True)

st.write("""
These metrics indicate that the model has a high predictive accuracy:
- The low **MSE** shows minimal prediction error, meaning the predicted batting averages are close to the actual values.
- The **R-squared value of 0.974** implies that the model explains 97.4% of the variance in batting average outcomes. This high R² suggests that the model captures the majority of necessary information, providing a strong indication of a well-fitting model.
""")

st.write("""
The **Random Forest Regressor** performed well because it effectively handles high-dimensional data by using random subsets of features for each tree, which helps reduce overfitting and improves generalization. Key features, such as **mlb_SLG** and **mlb_OBP**, were identified as strong predictors, aligning with known baseball statistics. With 100 estimators, the model balances accuracy and generalization, achieving both a high R² and low MSE.
""")

st.subheader("Future Work")
st.write("""
Moving forward, we plan to perform **cross-validation** to tune hyperparameters, which will help optimize model performance. Additionally, we want to explore other data preprocessing methods. Although we chose **standardization** due to the varied scale of features, we are considering **Principal Component Analysis (PCA)** as an alternative. This analysis will help us understand the impact on training time and model performance.
""")

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
    'Tasks Completed': ['Data Sourcing and Cleaning', 'Results and Discussion, Contribution Table, GANTT Chart', 'Streamlit + Github and Results', 'ML Model, Visualization/Metrics', 'Data Preprocessing']
}

df = pd.DataFrame(data)

# Display the table in Streamlit
st.title("Project Tasks Completed")
st.write(df)

st.header("References")

st.markdown(
    """
1. J. Zimmerman, “A new way to look at College Players’ Stats,” The Hardball Times,  https://tht.fangraphs.com/a-new-way-to-look-at-college-players-stats/ (accessed Oct. 4, 2024). 
2. “Baseball statistics, ” Baseball Reference, Feb. 23, 2024. https://www.baseball-reference.com/bullpen/Talk:Baseball_statistics
3. “Pythagorean Winning Percentage,” MLB Advanced Media, 2024. https://www.mlb.com/glossary/advanced-stats/pythagorean-winning-percentage
"""
)
