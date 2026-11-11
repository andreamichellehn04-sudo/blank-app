import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_gsheets import GSheetsConnection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Student Academic Performance Report")

conn = st.connection("gsheets", type=GSheetsConnection)

data = conn.read(
    spreadsheet="https://docs.google.com/spreadsheets/d/1h2Z07PVPRXjw-tH8e3Mz_EFeT2V4nspKRobHnWDuG2U/edit?usp=sharing",
    ttl=0
)

# Clean column names
data.columns = data.columns.str.strip().str.rstrip(":")

st.metric("Total Responses", len(data))
st.write(data.describe())
# =====================================================
# PREPARE VARIABLES
# =====================================================

age = "Age"
semester = "Semester"
career = "Career"

courses = "How many courses are you currently taking?"

performance = (
    "How would you describe your academic performance this semester "
    "compared with the previous semester?"
)

study_hours = (
    "On average, how many hours do you study per week outside of class?"
)

study_advance = "How often do you study in advance for your exams?"

final_grade = "What was your final grade the last semester?"

midterm_grade = (
    "In average, what were your grades during this mid term evaluations?"
)

sleep_hours = (
    "On average, how many hours of sleep do you get per night "
    "during mid - term or exams?"
)

sleep_quality = "How would you rate your overall sleep quality?"

stress = (
    "On a scale from 1 to 10, how would you rate your stress level "
    "during midterm exams?"
)


# =====================================================
# CONVERT NUMERICAL VARIABLES
# =====================================================

numeric_columns = [
    age,
    courses,
    study_hours,
    final_grade,
    midterm_grade,
    sleep_hours,
    stress
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")


# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

st.header("Descriptive Statistics")

st.dataframe(
    data[numeric_columns].describe().T
)


# =====================================================
# 1. STUDENT PROFILE
# =====================================================

st.header("1. Student Profile")


# AGE HISTOGRAM

fig_age = px.histogram(
    data,
    x=age,
    title="Age Distribution",
    nbins=10
)

st.plotly_chart(fig_age, use_container_width=True)


# AGE BOXPLOT

fig_age_box = px.box(
    data,
    y=age,
    points="all",
    title="Age Boxplot"
)

st.plotly_chart(fig_age_box, use_container_width=True)


# SEMESTER

semester_counts = (
    data[semester]
    .value_counts()
    .sort_index()
    .reset_index()
)

semester_counts.columns = ["Semester", "Frequency"]

fig_semester = px.bar(
    semester_counts,
    x="Semester",
    y="Frequency",
    title="Students by Semester",
    text_auto=True
)

st.plotly_chart(fig_semester, use_container_width=True)


# CAREER

career_counts = (
    data[career]
    .value_counts()
    .reset_index()
)

career_counts.columns = ["Career", "Frequency"]

fig_career = px.bar(
    career_counts,
    x="Career",
    y="Frequency",
    title="Students by Career",
    text_auto=True
)

st.plotly_chart(fig_career, use_container_width=True)


# =====================================================
# 2. ACADEMIC PERFORMANCE
# =====================================================

st.header("2. Academic Performance")


# NUMBER OF COURSES

course_counts = (
    data[courses]
    .value_counts()
    .sort_index()
    .reset_index()
)

course_counts.columns = ["Number of Courses", "Frequency"]

fig_courses = px.bar(
    course_counts,
    x="Number of Courses",
    y="Frequency",
    title="Number of Courses Currently Taken",
    text_auto=True
)

st.plotly_chart(fig_courses, use_container_width=True)


# PERFORMANCE COMPARED WITH PREVIOUS SEMESTER

performance_counts = (
    data[performance]
    .value_counts()
    .reset_index()
)

performance_counts.columns = ["Performance", "Frequency"]

fig_performance = px.bar(
    performance_counts,
    x="Performance",
    y="Frequency",
    title="Academic Performance Compared with Previous Semester",
    text_auto=True
)

st.plotly_chart(fig_performance, use_container_width=True)


# FINAL GRADE HISTOGRAM

fig_final_hist = px.histogram(
    data,
    x=final_grade,
    title="Final Grade Last Semester",
    nbins=10
)

st.plotly_chart(fig_final_hist, use_container_width=True)


# FINAL GRADE BOXPLOT

fig_final_box = px.box(
    data,
    y=final_grade,
    points="all",
    title="Final Grade Boxplot"
)

st.plotly_chart(fig_final_box, use_container_width=True)


# MIDTERM GRADE HISTOGRAM

fig_midterm_hist = px.histogram(
    data,
    x=midterm_grade,
    title="Midterm Grade Distribution",
    nbins=10
)

st.plotly_chart(fig_midterm_hist, use_container_width=True)


# MIDTERM GRADE BOXPLOT

fig_midterm_box = px.box(
    data,
    y=midterm_grade,
    points="all",
    title="Midterm Grade Boxplot"
)

st.plotly_chart(fig_midterm_box, use_container_width=True)


# =====================================================
# 3. STUDY HABITS
# =====================================================

st.header("3. Study Habits")


# STUDY HOURS HISTOGRAM

fig_study_hist = px.histogram(
    data,
    x=study_hours,
    title="Weekly Study Hours",
    nbins=10
)

st.plotly_chart(fig_study_hist, use_container_width=True)


# STUDY HOURS BOXPLOT

fig_study_box = px.box(
    data,
    y=study_hours,
    points="all",
    title="Weekly Study Hours Boxplot"
)

st.plotly_chart(fig_study_box, use_container_width=True)


# STUDY IN ADVANCE

study_counts = (
    data[study_advance]
    .value_counts()
    .reset_index()
)

study_counts.columns = ["Study Frequency", "Frequency"]

fig_study = px.bar(
    study_counts,
    x="Study Frequency",
    y="Frequency",
    title="How Often Students Study in Advance",
    text_auto=True
)

st.plotly_chart(fig_study, use_container_width=True)


# =====================================================
# 4. SLEEP AND STRESS
# =====================================================

st.header("4. Sleep and Stress")


# SLEEP HOURS HISTOGRAM

fig_sleep_hist = px.histogram(
    data,
    x=sleep_hours,
    title="Hours of Sleep During Exams",
    nbins=10
)

st.plotly_chart(fig_sleep_hist, use_container_width=True)


# SLEEP HOURS BOXPLOT

fig_sleep_box = px.box(
    data,
    y=sleep_hours,
    points="all",
    title="Sleep Hours Boxplot"
)

st.plotly_chart(fig_sleep_box, use_container_width=True)


# SLEEP QUALITY

sleep_counts = (
    data[sleep_quality]
    .value_counts()
    .reset_index()
)

sleep_counts.columns = ["Sleep Quality", "Frequency"]

fig_sleep_quality = px.bar(
    sleep_counts,
    x="Sleep Quality",
    y="Frequency",
    title="Overall Sleep Quality",
    text_auto=True
)

st.plotly_chart(fig_sleep_quality, use_container_width=True)


# STRESS LEVEL

stress_counts = (
    data[stress]
    .value_counts()
    .sort_index()
    .reset_index()
)

stress_counts.columns = ["Stress Level", "Frequency"]

fig_stress = px.bar(
    stress_counts,
    x="Stress Level",
    y="Frequency",
    title="Stress Level During Midterm Exams",
    text_auto=True
)

st.plotly_chart(fig_stress, use_container_width=True)
# 5. RANDOM FOREST REGRESSION
# =====================================================

st.header("5. Machine Learning - Random Forest Regression")

st.write(
    "This model predicts students' midterm grades using "
    "academic habits, sleep, stress, and previous academic performance."
)

# -----------------------------------------------------
# VARIABLES USED IN THE MODEL
# -----------------------------------------------------

features = [
    age,
    courses,
    study_hours,
    final_grade,
    sleep_hours,
    stress
]

target = midterm_grade


# -----------------------------------------------------
# PREPARE DATA
# -----------------------------------------------------

ml_data = data[features + [target]].copy()

# Convert everything to numeric
for column in features + [target]:
    ml_data[column] = pd.to_numeric(
        ml_data[column],
        errors="coerce"
    )

# Remove rows with missing information
ml_data = ml_data.dropna()


# Show how many responses are being used
st.write("Responses used in the model:", len(ml_data))


# -----------------------------------------------------
# CHECK SAMPLE SIZE
# -----------------------------------------------------

if len(ml_data) < 10:

    st.warning(
        "There are not enough complete responses to train "
        "the model reliably. More responses are recommended."
    )

else:

    # -------------------------------------------------
    # X AND y
    # -------------------------------------------------

    X = ml_data[features]

    y = ml_data[target]


    # -------------------------------------------------
    # TRAIN AND TEST DATA
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


    # -------------------------------------------------
    # RANDOM FOREST MODEL
    # -------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)


    # -------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------

    predictions = model.predict(X_test)


    # -------------------------------------------------
    # MODEL RESULTS
    # -------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:.2f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    col3.metric(
        "R²",
        f"{r2:.2f}"
    )


    # -------------------------------------------------
    # ACTUAL VS PREDICTED VALUES
    # -------------------------------------------------

    results = pd.DataFrame({
        "Actual Grade": y_test.values,
        "Predicted Grade": predictions
    })

    results["Absolute Error"] = abs(
        results["Actual Grade"]
        - results["Predicted Grade"]
    )

    st.subheader("Actual vs Predicted Grades")

    st.dataframe(
        results,
        use_container_width=True
    )


    # -------------------------------------------------
    # GRAPH: ACTUAL VS PREDICTED
    # -------------------------------------------------

    fig_predictions = px.scatter(
        results,
        x="Actual Grade",
        y="Predicted Grade",
        title="Actual vs Predicted Grades"
    )

    minimum = min(
        results["Actual Grade"].min(),
        results["Predicted Grade"].min()
    )

    maximum = max(
        results["Actual Grade"].max(),
        results["Predicted Grade"].max()
    )

    fig_predictions.add_shape(
        type="line",
        x0=minimum,
        y0=minimum,
        x1=maximum,
        y1=maximum,
        line=dict(dash="dash")
    )

    st.plotly_chart(
        fig_predictions,
        use_container_width=True
    )


    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------

    st.subheader("Most Important Variables")

    importance = pd.DataFrame({
        "Variable": features,
        "Importance": model.feature_importances_
    })

    importance["Importance (%)"] = (
        importance["Importance"] * 100
    )

    importance = importance.sort_values(
        by="Importance (%)",
        ascending=False
    )

    st.dataframe(
        importance,
        use_container_width=True
    )

    fig_importance = px.bar(
        importance,
        x="Importance (%)",
        y="Variable",
        orientation="h",
        title="Feature Importance - Random Forest"
    )

    fig_importance.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )


    # -------------------------------------------------
    # TRY A NEW PREDICTION
    # -------------------------------------------------

    st.subheader("Predict a Student's Grade")

    st.write(
        "Enter the characteristics of a student "
        "to obtain a predicted midterm grade."
    )

    new_age = st.number_input(
        "Age",
        value=float(ml_data[age].median())
    )

    new_courses = st.number_input(
        "Number of courses",
        value=float(ml_data[courses].median())
    )

    new_study_hours = st.number_input(
        "Study hours per week",
        value=float(ml_data[study_hours].median())
    )

    new_previous_grade = st.number_input(
        "Final grade last semester",
        value=float(ml_data[final_grade].median())
    )

    new_sleep = st.number_input(
        "Hours of sleep",
        value=float(ml_data[sleep_hours].median())
    )

    new_stress = st.number_input(
        "Stress level",
        value=float(ml_data[stress].median())
    )


    new_student = pd.DataFrame({
        age: [new_age],
        courses: [new_courses],
        study_hours: [new_study_hours],
        final_grade: [new_previous_grade],
        sleep_hours: [new_sleep],
        stress: [new_stress]
    })


    predicted_grade = model.predict(
        new_student
    )[0]


    st.metric(
        "Predicted Midterm Grade",
        f"{predicted_grade:.2f}"
    )
    