import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_gsheets import GSheetsConnection

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