import streamlit as st
import pandas as pd
import numpy as np
import re

from pypdf import PdfReader

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Career & Salary Predictor",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("💼 AI Career & Salary Recommendation System")

st.write(
    "Upload your resume to predict suitable job opportunities, "
    "estimate your salary, and identify skills you should learn "
    "to improve your career prospects."
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "Employee_Salary_Career_Synthetic_Dataset.csv"
    )


df = load_dataset()


# ============================================================
# EDUCATION MAPPING
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


# ============================================================
# MODEL FEATURES
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


# Skills that will be used for skill-gap analysis
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
# PREPARE DATA
# ============================================================

df_model = df.copy()

df_model["Education"] = df_model["Education"].map(
    education_mapping
)

X = df_model[features]

y_job = df_model["Job_Role"]

y_salary = df_model["Salary_LPA"]


# ============================================================
# TRAIN JOB CLASSIFICATION MODEL
# ============================================================

@st.cache_resource
def train_job_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy


job_model, job_accuracy = train_job_model(
    X,
    y_job
)


# ============================================================
# TRAIN SALARY REGRESSION MODEL
# ============================================================

@st.cache_resource
def train_salary_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return model, mae, r2


salary_model, salary_mae, salary_r2 = train_salary_model(
    X,
    y_salary
)


# ============================================================
# RESUME SKILL KEYWORDS
# ============================================================

skill_keywords = {

    "Programming": [
        "programming",
        "software development",
        "coding",
        "c programming",
        "c++",
        "java",
        "javascript",
        "typescript",
        "golang",
        "rust"
    ],

    "Python": [
        "python",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "django",
        "flask"
    ],

    "SQL": [
        "sql",
        "mysql",
        "postgresql",
        "oracle",
        "database",
        "mongodb"
    ],

    "Machine_Learning": [
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "sklearn",
        "natural language processing",
        "nlp",
        "computer vision"
    ],

    "Cloud": [
        "cloud",
        "aws",
        "amazon web services",
        "azure",
        "google cloud",
        "gcp",
        "docker",
        "kubernetes"
    ],

    "Communication": [
        "communication",
        "presentation",
        "public speaking",
        "teamwork",
        "interpersonal skills",
        "collaboration"
    ],

    "Leadership": [
        "leadership",
        "team lead",
        "team leader",
        "managed team",
        "management",
        "supervised"
    ],

    "Data_Analysis": [
        "data analysis",
        "data analytics",
        "excel",
        "power bi",
        "tableau",
        "statistics",
        "data visualization",
        "pandas"
    ],

    "Electronics": [
        "electronics",
        "embedded systems",
        "microcontroller",
        "microprocessor",
        "arduino",
        "esp32",
        "raspberry pi",
        "pcb",
        "circuit",
        "vlsi",
        "verilog",
        "fpga",
        "sensors",
        "embedded c"
    ],

    "Networking": [
        "networking",
        "computer networks",
        "tcp/ip",
        "tcp",
        "udp",
        "routing",
        "switching",
        "cisco",
        "firewall",
        "network security"
    ],

    "Linux": [
        "linux",
        "ubuntu",
        "unix",
        "bash",
        "shell scripting"
    ]
}


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# ============================================================
# TXT TEXT EXTRACTION
# ============================================================

def extract_txt_text(uploaded_file):

    return uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )


# ============================================================
# DETECT SKILLS
# ============================================================

def detect_skills(text):

    text = text.lower()

    detected = {}

    for skill, keywords in skill_keywords.items():

        matches = 0

        for keyword in keywords:

            if keyword.lower() in text:

                matches += 1

        detected[skill] = matches

    return detected


# ============================================================
# CONVERT KEYWORD MATCHES TO SKILL SCORE
# ============================================================

def calculate_skill_scores(detected):

    scores = {}

    for skill, count in detected.items():

        if count == 0:

            scores[skill] = 1

        elif count == 1:

            scores[skill] = 4

        elif count == 2:

            scores[skill] = 6

        elif count == 3:

            scores[skill] = 8

        else:

            scores[skill] = 10

    return scores


# ============================================================
# EXTRACT EXPERIENCE FROM RESUME
# ============================================================

def extract_experience(text):

    text_lower = text.lower()

    patterns = [

        r"(\d+(?:\.\d+)?)\+?\s*(?:years|year)\s*(?:of)?\s*experience",

        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:years|year)"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            try:

                return float(
                    match.group(1)
                )

            except ValueError:

                pass

    return 1.0


# ============================================================
# DETECT EDUCATION
# ============================================================

def detect_education(text):

    text = text.lower()

    if "m.tech" in text or "mtech" in text:

        return 4

    if "m.sc" in text or "msc" in text:

        return 3

    if "mba" in text:

        return 4

    if "b.tech" in text or "btech" in text:

        return 2

    if "bca" in text:

        return 1

    if "b.sc" in text or "bsc" in text:

        return 1

    if "diploma" in text:

        return 0

    # Default assumption
    return 2


# ============================================================
# ESTIMATE NUMBER OF PROJECTS
# ============================================================

def estimate_projects(text):

    text_lower = text.lower()

    project_keywords = [
        "project",
        "developed",
        "built",
        "implemented",
        "created",
        "designed"
    ]

    count = 0

    for keyword in project_keywords:

        count += text_lower.count(
            keyword
        )

    return int(
        np.clip(
            count // 2,
            0,
            10
        )
    )


# ============================================================
# ESTIMATE CERTIFICATIONS
# ============================================================

def estimate_certifications(text):

    text_lower = text.lower()

    keywords = [
        "certification",
        "certified",
        "certificate",
        "aws certified",
        "azure certified",
        "coursera",
        "udemy"
    ]

    count = 0

    for keyword in keywords:

        count += text_lower.count(
            keyword
        )

    return int(
        np.clip(
            count,
            0,
            8
        )
    )


# ============================================================
# CREATE PROFILE FROM RESUME
# ============================================================

def create_resume_profile(text):

    detected = detect_skills(
        text
    )

    skill_scores = calculate_skill_scores(
        detected
    )

    experience = extract_experience(
        text
    )

    education = detect_education(
        text
    )

    projects = estimate_projects(
        text
    )

    certifications = estimate_certifications(
        text
    )

    profile = {

        "Education": education,

        "Experience_Years": experience,

        "Programming":
            skill_scores["Programming"],

        "Python":
            skill_scores["Python"],

        "SQL":
            skill_scores["SQL"],

        "Machine_Learning":
            skill_scores["Machine_Learning"],

        "Cloud":
            skill_scores["Cloud"],

        "Communication":
            skill_scores["Communication"],

        "Leadership":
            skill_scores["Leadership"],

        "Data_Analysis":
            skill_scores["Data_Analysis"],

        "Electronics":
            skill_scores["Electronics"],

        "Networking":
            skill_scores["Networking"],

        "Linux":
            skill_scores["Linux"],

        "Project_Count":
            projects,

        "Certifications":
            certifications
    }

    return profile, detected


# ============================================================
# MANUAL INPUT
# ============================================================

def manual_profile():

    st.subheader(
        "Enter Your Information"
    )

    education = st.selectbox(
        "Education",
        [
            "Diploma",
            "BCA",
            "B.Sc",
            "B.Tech",
            "M.Sc",
            "M.Tech",
            "MBA"
        ]
    )

    experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=12.0,
        value=1.0,
        step=0.5
    )

    profile = {

        "Education":
            education_mapping[
                education
            ],

        "Experience_Years":
            experience
    }

    for skill in skill_columns:

        profile[skill] = st.slider(
            skill.replace(
                "_",
                " "
            ),
            min_value=1,
            max_value=10,
            value=5
        )

    profile["Project_Count"] = st.number_input(
        "Number of Projects",
        min_value=0,
        max_value=15,
        value=2
    )

    profile["Certifications"] = st.number_input(
        "Number of Certifications",
        min_value=0,
        max_value=8,
        value=1
    )

    return profile


# ============================================================
# SELECT INPUT METHOD
# ============================================================

st.subheader(
    "📄 Choose Input Method"
)

input_method = st.radio(
    "How would you like to provide your information?",
    [
        "Upload Resume",
        "Enter Skills Manually"
    ],
    horizontal=True
)


profile = None
detected = None


# ============================================================
# RESUME UPLOAD
# ============================================================

if input_method == "Upload Resume":

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "txt"],
        help="Upload a text-based PDF or TXT resume."
    )

    if uploaded_file is not None:

        try:

            # --------------------------------------------
            # Extract text
            # --------------------------------------------

            if uploaded_file.name.lower().endswith(
                ".pdf"
            ):

                resume_text = extract_pdf_text(
                    uploaded_file
                )

            else:

                resume_text = extract_txt_text(
                    uploaded_file
                )


            # --------------------------------------------
            # Check extracted text
            # --------------------------------------------

            if not resume_text.strip():

                st.error(
                    "Could not extract text from this resume."
                )

                st.info(
                    "If this is a scanned/image-only PDF, "
                    "text extraction may not work."
                )

            else:

                st.success(
                    "Resume successfully processed!"
                )


                # ----------------------------------------
                # Show extracted text
                # ----------------------------------------

                with st.expander(
                    "View extracted resume text"
                ):

                    st.text(
                        resume_text[:5000]
                    )


                # ----------------------------------------
                # Create profile
                # ----------------------------------------

                profile, detected = create_resume_profile(
                    resume_text
                )


                # ----------------------------------------
                # Display detected skills
                # ----------------------------------------

                st.subheader(
                    "🔍 Skills Detected From Resume"
                )

                detected_display = {}

                for skill, count in detected.items():

                    if count > 0:

                        detected_display[
                            skill.replace(
                                "_",
                                " "
                            )
                        ] = count


                if detected_display:

                    st.write(
                        detected_display
                    )

                else:

                    st.warning(
                        "No recognized skills were found. "
                        "Try uploading a resume containing "
                        "technical skills."
                    )


                # ----------------------------------------
                # Show extracted profile
                # ----------------------------------------

                with st.expander(
                    "View extracted profile"
                ):

                    profile_display = {}

                    for key, value in profile.items():

                        profile_display[
                            key.replace(
                                "_",
                                " "
                            )
                        ] = value

                    st.dataframe(
                        pd.DataFrame(
                            [profile_display]
                        ),
                        use_container_width=True,
                        hide_index=True
                    )


        except Exception as e:

            st.error(
                f"Error processing resume: {e}"
            )


# ============================================================
# MANUAL INPUT
# ============================================================

else:

    profile = manual_profile()


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🚀 Analyze Career",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze:

    if profile is None:

        st.warning(
            "Please upload a resume before analyzing."
        )

    else:

        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        profile_df = pd.DataFrame(
            [profile],
            columns=features
        )


        # ====================================================
        # JOB PREDICTION
        # ====================================================

        predicted_job = job_model.predict(
            profile_df
        )[0]

        probabilities = job_model.predict_proba(
            profile_df
        )[0]

        confidence = (
            max(probabilities) * 100
        )


        # ====================================================
        # SALARY PREDICTION
        # ====================================================

        predicted_salary = salary_model.predict(
            profile_df
        )[0]


        # ====================================================
        # CAREER ANALYSIS
        # ====================================================

        st.divider()

        st.header(
            "🎯 Career Analysis"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Recommended Job",
                predicted_job
            )


        with col2:

            st.metric(
                "Estimated Salary",
                f"₹{predicted_salary:.2f} LPA"
            )


        with col3:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.1f}%"
            )


        # ====================================================
        # JOB PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Job Opportunities"
        )

        probability_df = pd.DataFrame({

            "Job Role":
                job_model.classes_,

            "Probability":
                probabilities * 100

        })


        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )


        probability_df[
            "Probability"
        ] = probability_df[
            "Probability"
        ].round(2)


        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


        st.bar_chart(
            probability_df.set_index(
                "Job Role"
            )["Probability"]
        )


        # ====================================================
        # HIGHER-PAYING JOBS
        # ====================================================

        st.subheader(
            "📈 Potential Higher-Paying Careers"
        )


        average_salaries = df.groupby(
            "Job_Role"
        )["Salary_LPA"].mean().sort_values(
            ascending=False
        )


        current_average_salary = average_salaries[
            predicted_job
        ]


        better_jobs = average_salaries[
            average_salaries >
            current_average_salary
        ]


        if len(better_jobs) == 0:

            st.info(
                "Your predicted career is already "
                "among the highest-paying categories "
                "in this dataset."
            )

        else:

            higher_job_data = []

            for job, salary in better_jobs.items():

                higher_job_data.append({

                    "Potential Job":
                        job,

                    "Average Salary":
                        f"₹{salary:.2f} LPA"

                })


            st.dataframe(
                pd.DataFrame(
                    higher_job_data
                ),
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # TARGET CAREER + SKILL GAP
        # ====================================================

        if len(better_jobs) > 0:

            # Select the closest higher-paying
            # career as the next target

            target_job = better_jobs.index[-1]

            target_salary = better_jobs.iloc[-1]


            st.subheader(
                "🚀 Recommended Next Career"
            )


            target_col1, target_col2 = st.columns(2)


            with target_col1:

                st.success(
                    target_job
                )


            with target_col2:

                st.metric(
                    "Average Salary",
                    f"₹{target_salary:.2f} LPA"
                )


            # =================================================
            # SKILL PROFILES
            # =================================================

            job_skill_profiles = df.groupby(
                "Job_Role"
            )[skill_columns].mean()


            target_profile = job_skill_profiles.loc[
                target_job
            ]


            # =================================================
            # CALCULATE SKILL GAPS
            # =================================================

            skill_gaps = []


            for skill in skill_columns:

                current_level = profile[
                    skill
                ]

                required_level = target_profile[
                    skill
                ]

                gap = (
                    required_level -
                    current_level
                )


                if gap >= 1.5:

                    skill_gaps.append({

                        "Skill":
                            skill.replace(
                                "_",
                                " "
                            ),

                        "Current Level":
                            round(
                                current_level,
                                1
                            ),

                        "Target Level":
                            round(
                                required_level,
                                1
                            ),

                        "Skill Gap":
                            round(
                                gap,
                                1
                            )

                    })


            # =================================================
            # DISPLAY SKILL GAPS
            # =================================================

            st.subheader(
                "📚 Skills You Should Learn"
            )


            if len(skill_gaps) > 0:

                skill_gap_df = pd.DataFrame(
                    skill_gaps
                )


                skill_gap_df = skill_gap_df.sort_values(
                    "Skill Gap",
                    ascending=False
                )


                st.dataframe(
                    skill_gap_df.head(5),
                    use_container_width=True,
                    hide_index=True
                )


                st.info(
                    "These are the skills with the largest "
                    "gap between your current profile and "
                    "the average skill profile of the "
                    "recommended target career."
                )


            else:

                st.success(
                    "Your current skill profile is already "
                    "close to the requirements of the "
                    "recommended target career."
                )


        # ====================================================
        # CURRENT SKILL PROFILE
        # ====================================================

        st.subheader(
            "📊 Your Skill Profile"
        )


        current_skill_data = []


        for skill in skill_columns:

            current_skill_data.append({

                "Skill":
                    skill.replace(
                        "_",
                        " "
                    ),

                "Score":
                    profile[skill]

            })


        current_skill_df = pd.DataFrame(
            current_skill_data
        )


        st.bar_chart(
            current_skill_df.set_index(
                "Skill"
            )["Score"]
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        with st.expander(
            "ℹ️ Model Information"
        ):

            st.write(
                f"**Job Classification Accuracy:** "
                f"{job_accuracy * 100:.2f}%"
            )

            st.write(
                f"**Salary Model MAE:** "
                f"{salary_mae:.2f} LPA"
            )

            st.write(
                f"**Salary Model R²:** "
                f"{salary_r2:.2f}"
            )

            st.write(
                "**Job Model:** Random Forest Classifier"
            )

            st.write(
                "**Salary Model:** Random Forest Regressor"
            )

            st.write(
                "**Dataset:** 1,000 synthetic employee records"
            )

            st.write(
                "**Resume Processing:** "
                "Keyword-based skill extraction"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Academic ML Mini Project | "
    "Predictions are based on a synthetic dataset "
    "and should not be treated as real-world salary estimates."
)
