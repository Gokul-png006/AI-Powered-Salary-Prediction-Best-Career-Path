import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# AI EMPLOYEE SALARY, JOB & SKILL RECOMMENDATION SYSTEM
# ============================================================

print("=" * 60)
print(" AI EMPLOYEE SALARY & CAREER RECOMMENDATION SYSTEM")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Employee_Salary_Career_Synthetic_Dataset.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nJob Distribution:")
print(df["Job_Role"].value_counts())


# ============================================================
# 2. PREPROCESS EDUCATION
# ============================================================

education_mapping = {
    "Diploma": 0,
    "BCA": 1,
    "B.Sc": 1,
    "B.Tech": 2,
    "M.Sc": 3,
    "M.Tech": 4,
    "MBA": 4
}

df["Education"] = df["Education"].map(education_mapping)


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

features = [
    "Education",
    "Experience_Years",
    "Programming",
    "Python",
    "SQL",
    "Machine_Learning",
    "Cloud",
    "Communication",
    "Leadership",
    "Data_Analysis",
    "Electronics",
    "Networking",
    "Linux",
    "Project_Count",
    "Certifications"
]

X = df[features]


# ============================================================
# 4. JOB ROLE CLASSIFICATION
# ============================================================

print("\n")
print("=" * 60)
print(" JOB ROLE PREDICTION")
print("=" * 60)

y_job = df["Job_Role"]

X_train_job, X_test_job, y_train_job, y_test_job = train_test_split(
    X,
    y_job,
    test_size=0.20,
    random_state=42,
    stratify=y_job
)

job_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

job_model.fit(
    X_train_job,
    y_train_job
)

job_predictions = job_model.predict(
    X_test_job
)

job_accuracy = accuracy_score(
    y_test_job,
    job_predictions
)

print("\nJob Prediction Accuracy:")
print(f"{job_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test_job,
        job_predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test_job,
        job_predictions
    )
)


# ============================================================
# 5. SALARY REGRESSION
# ============================================================

print("\n")
print("=" * 60)
print(" SALARY PREDICTION")
print("=" * 60)

y_salary = df["Salary_LPA"]

X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(
    X,
    y_salary,
    test_size=0.20,
    random_state=42
)

salary_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

salary_model.fit(
    X_train_salary,
    y_train_salary
)

salary_predictions = salary_model.predict(
    X_test_salary
)


# ============================================================
# 6. SALARY MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test_salary,
    salary_predictions
)

mse = mean_squared_error(
    y_test_salary,
    salary_predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test_salary,
    salary_predictions
)

print("\nMean Absolute Error:")
print(f"{mae:.2f} LPA")

print("\nRoot Mean Squared Error:")
print(f"{rmse:.2f} LPA")

print("\nR² Score:")
print(f"{r2:.2f}")


# ============================================================
# 7. EXAMPLE EMPLOYEE
# ============================================================

print("\n")
print("=" * 60)
print(" EXAMPLE EMPLOYEE")
print("=" * 60)

example_employee = pd.DataFrame([[
    2,      # Education = B.Tech
    2.5,    # Experience
    8,      # Programming
    8,      # Python
    7,      # SQL
    7,      # Machine Learning
    5,      # Cloud
    7,      # Communication
    4,      # Leadership
    7,      # Data Analysis
    3,      # Electronics
    4,      # Networking
    5,      # Linux
    4,      # Projects
    2       # Certifications
]], columns=features)


# ============================================================
# 8. PREDICT JOB
# ============================================================

predicted_job = job_model.predict(
    example_employee
)[0]

job_probabilities = job_model.predict_proba(
    example_employee
)[0]

job_classes = job_model.classes_

job_confidence = max(
    job_probabilities
) * 100


# ============================================================
# 9. PREDICT SALARY
# ============================================================

predicted_salary = salary_model.predict(
    example_employee
)[0]


print("\nPredicted Job:")
print(predicted_job)

print("\nJob Prediction Confidence:")
print(f"{job_confidence:.2f}%")

print("\nPredicted Salary:")
print(f"{predicted_salary:.2f} LPA")


# ============================================================
# 10. JOB PROBABILITIES
# ============================================================

print("\n")
print("Job Probabilities:")

for job, probability in zip(
    job_classes,
    job_probabilities
):

    print(
        f"{job}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# 11. SKILL GAP ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print(" SKILL IMPROVEMENT RECOMMENDATION")
print("=" * 60)


# Skills used for comparison

skill_columns = [
    "Programming",
    "Python",
    "SQL",
    "Machine_Learning",
    "Cloud",
    "Communication",
    "Leadership",
    "Data_Analysis",
    "Electronics",
    "Networking",
    "Linux"
]


# ============================================================
# 12. CURRENT USER SKILLS
# ============================================================

user_skills = {
    "Programming": 8,
    "Python": 8,
    "SQL": 7,
    "Machine_Learning": 7,
    "Cloud": 5,
    "Communication": 7,
    "Leadership": 4,
    "Data_Analysis": 7,
    "Electronics": 3,
    "Networking": 4,
    "Linux": 5
}


# ============================================================
# 13. CALCULATE AVERAGE SKILLS FOR EACH JOB
# ============================================================

job_skill_profiles = df.groupby(
    "Job_Role"
)[skill_columns].mean()


# ============================================================
# 14. CURRENT JOB AVERAGE SALARY
# ============================================================

current_job_salary = df[
    df["Job_Role"] == predicted_job
]["Salary_LPA"].mean()


print("\nCurrent Predicted Job:")
print(predicted_job)

print(
    f"\nAverage Salary for Current Job: "
    f"{current_job_salary:.2f} LPA"
)


# ============================================================
# 15. FIND HIGHER-PAYING JOBS
# ============================================================

job_salary_comparison = df.groupby(
    "Job_Role"
)["Salary_LPA"].mean().sort_values(
    ascending=False
)

better_jobs = job_salary_comparison[
    job_salary_comparison > current_job_salary
]

print("\nPotential Higher-Paying Jobs:")

if len(better_jobs) == 0:

    print(
        "No higher-paying job categories "
        "were found in the dataset."
    )

else:

    for job, salary in better_jobs.items():

        print(
            f"{job}: "
            f"{salary:.2f} LPA"
        )


# ============================================================
# 16. SELECT TARGET JOB
# ============================================================

if len(better_jobs) > 0:

    # Choose the lowest higher-paying job as the
    # most realistic next step

    target_job = better_jobs.index[-1]

    target_salary = better_jobs.iloc[-1]

    print("\n")
    print("Recommended Target Job:")
    print(target_job)

    print(
        f"Average Salary: "
        f"{target_salary:.2f} LPA"
    )


    # ========================================================
    # 17. COMPARE CURRENT SKILLS WITH TARGET JOB
    # ========================================================

    target_profile = job_skill_profiles.loc[
        target_job
    ]

    skill_gaps = []

    for skill in skill_columns:

        current_level = user_skills[
            skill
        ]

        required_level = target_profile[
            skill
        ]

        gap = required_level - current_level

        if gap >= 1.5:

            skill_gaps.append(
                (
                    skill,
                    current_level,
                    round(required_level, 1),
                    round(gap, 1)
                )
            )


    # ========================================================
    # 18. SORT SKILL GAPS
    # ========================================================

    skill_gaps.sort(
        key=lambda x: x[3],
        reverse=True
    )


    # ========================================================
    # 19. DISPLAY SKILLS TO LEARN
    # ========================================================

    print("\n")
    print("Skills You Should Learn/Improve:")

    if len(skill_gaps) > 0:

        for (
            skill,
            current,
            required,
            gap
        ) in skill_gaps[:5]:

            print(
                f"\n{skill}"
            )

            print(
                f"Current Level : {current}/10"
            )

            print(
                f"Target Level  : {required}/10"
            )

            print(
                f"Skill Gap     : {gap}"
            )

    else:

        print(
            "Your current skills are already "
            "close to the target job requirements."
        )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print(" FINAL CAREER REPORT")
print("=" * 60)

print(
    f"\nRecommended Job: "
    f"{predicted_job}"
)

print(
    f"Job Confidence: "
    f"{job_confidence:.2f}%"
)

print(
    f"Predicted Salary: "
    f"{predicted_salary:.2f} LPA"
)

if len(better_jobs) > 0:

    print(
        f"Potential Next Job: "
        f"{target_job}"
    )

    print(
        "\nTop Skills to Improve:"
    )

    if len(skill_gaps) > 0:

        for item in skill_gaps[:5]:

            print(
                f"- {item[0]}"
            )

else:

    print(
        "\nNo higher-paying target job "
        "identified."
    )

print("\n")
print("=" * 60)
print(" END OF REPORT")
print("=" * 60)
