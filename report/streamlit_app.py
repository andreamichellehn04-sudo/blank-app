import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from streamlit_gsheets import GSheetsConnection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Student Academic Performance",
    page_icon="🎓",
    layout="wide"
)

PRIMARY = "#4F46E5"
ACCENT = "#F59E0B"
PINK = "#EC4899"
SLATE = "#475569"

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #E2E8F0;
        }

        .hypothesis-box {
            background: linear-gradient(135deg, #EEF2FF 0%, #ECFEFF 100%);
            border: 2px solid #818CF8;
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 18px;
            color: #1E293B !important;
        }

        .hypothesis-box b {
            color: #3730A3 !important;
        }

        .hypothesis-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #312E81 !important;
            margin-bottom: 15px;
        }

        [data-testid="stMetric"] {
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            padding: 18px;
            border-radius: 14px;
        }

        [data-testid="stMetricLabel"] {
            color: #475569 !important;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #1E3A8A !important;
            font-weight: 700;
        }

        [data-testid="stMetricDelta"] {
            color: #475569 !important;
        }

        button[data-baseweb="tab"] {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def style_fig(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


st.title("🎓 Student Academic Performance Dashboard")
st.caption(
    "Exploratory analysis and Random Forest Regression based on university student survey responses."
)

# IMPORTANT: HTML starts at the left edge so Streamlit does not render it as code.
st.markdown(
    """
<div class="hypothesis-box">
<div class="hypothesis-title">Research Hypothesis</div>
<b>H₀:</b> Study hours, sleep hours, and previous grades do not provide relevant information for predicting students' academic performance this semester.
<br><br>
<b>H₁:</b> Study hours, sleep hours, and previous grades provide relevant information for predicting students' academic performance this semester.
</div>
""",
    unsafe_allow_html=True
)

conn = st.connection("gsheets", type=GSheetsConnection)

data = conn.read(
    spreadsheet="https://docs.google.com/spreadsheets/d/1h2Z07PVPRXjw-tH8e3Mz_EFeT2V4nspKRobHnWDuG2U/edit?usp=sharing",
    ttl=0
)

data.columns = data.columns.str.strip().str.rstrip(":")

age = "Age"
semester = "Semester"
career = "Career"
courses = "How many courses are you currently taking?"
performance = (
    "How would you describe your academic performance this semester "
    "compared with the previous semester?"
)
study_hours = "On average, how many hours do you study per week outside of class?"
study_advance = "How often do you study in advance for your exams?"
final_grade = "What was your final grade the last semester?"
midterm_grade = "In average, what were your grades during this mid term evaluations?"
sleep_hours = (
    "On average, how many hours of sleep do you get per night "
    "during mid - term or exams?"
)
sleep_quality = "How would you rate your overall sleep quality?"
stress = (
    "On a scale from 1 to 10, how would you rate your stress level "
    "during midterm exams?"
)

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
# DATA CLEANING AND OUTLIER REMOVAL
# =====================================================

original_responses = len(data)


# -----------------------------------------------------
# 1. STANDARDIZE GRADES TO A 0-10 SCALE
# -----------------------------------------------------

def normalize_grade(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    # If someone entered 80, 90, 100 instead of 8, 9, 10
    series = series.apply(
        lambda x: x / 10
        if pd.notna(x) and 10 < x <= 100
        else x
    )

    # Values outside the valid 0-10 range
    # are treated as invalid
    series = series.where(
        series.between(0, 10)
    )

    return series


data[final_grade] = normalize_grade(
    data[final_grade]
)

data[midterm_grade] = normalize_grade(
    data[midterm_grade]
)


# -----------------------------------------------------
# 2. REMOVE IMPOSSIBLE VALUES
# -----------------------------------------------------

valid_ranges = {

    age: (17, 60),

    courses: (1, 15),

    study_hours: (0, 60),

    final_grade: (0, 10),

    midterm_grade: (0, 10),

    sleep_hours: (0, 12),

    stress: (1, 10)

}


valid_rows = pd.Series(
    True,
    index=data.index
)


for column, (minimum, maximum) in valid_ranges.items():

    valid_rows &= (
        data[column].isna()
        |
        data[column].between(
            minimum,
            maximum
        )
    )


data = data[
    valid_rows
].copy()


# -----------------------------------------------------
# 3. IQR METHOD FOR EXTREME VALUES
# -----------------------------------------------------

def remove_outliers_iqr(
    dataframe,
    columns
):

    cleaned = dataframe.copy()

    for column in columns:

        values = cleaned[
            column
        ].dropna()

        if len(values) < 4:
            continue

        Q1 = values.quantile(0.25)

        Q3 = values.quantile(0.75)

        IQR = Q3 - Q1


        lower_limit = (
            Q1 - 1.5 * IQR
        )

        upper_limit = (
            Q3 + 1.5 * IQR
        )


        cleaned = cleaned[
            cleaned[column].isna()
            |
            cleaned[column].between(
                lower_limit,
                upper_limit
            )
        ]

    return cleaned


# Apply IQR only to variables where
# extreme responses can distort the analysis

data = remove_outliers_iqr(

    data,

    [
        study_hours,
        sleep_hours
    ]

)


# -----------------------------------------------------
# CLEANING SUMMARY
# -----------------------------------------------------

clean_responses = len(data)

removed_responses = (
    original_responses
    - clean_responses
)


tab1, tab2, tab3 = st.tabs(
    ["📊 Overview", "🤖 Machine Learning", "🎯 Prediction Tool"]
)

with tab1:
    st.subheader("Survey Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Responses", len(data))
    c2.metric(
        "Average Midterm Grade",
        f"{data[midterm_grade].mean():.2f}" if data[midterm_grade].notna().any() else "N/A"
    )
    c3.metric(
        "Average Study Hours",
        f"{data[study_hours].mean():.1f} hrs" if data[study_hours].notna().any() else "N/A"
    )
    c4.metric(
        "Average Sleep",
        f"{data[sleep_hours].mean():.1f} hrs" if data[sleep_hours].notna().any() else "N/A"
    )

    st.divider()
    st.subheader("Key Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig_midterm = px.histogram(
            data,
            x=midterm_grade,
            nbins=10,
            marginal="box",
            title="Midterm Grade Distribution",
            color_discrete_sequence=[PRIMARY]
        )
        fig_midterm.update_layout(
            xaxis_title="Midterm Grade",
            yaxis_title="Students",
            showlegend=False
        )
        st.plotly_chart(style_fig(fig_midterm), use_container_width=True)

    with col2:
        fig_previous = px.scatter(
            data,
            x=final_grade,
            y=midterm_grade,
            color=stress,
            color_continuous_scale="Viridis",
            title="Previous Grade vs Current Midterm Grade",
            labels={
                final_grade: "Previous Grade",
                midterm_grade: "Current Midterm Grade",
                stress: "Stress"
            }
        )
        st.plotly_chart(style_fig(fig_previous), use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig_study = px.scatter(
            data,
            x=study_hours,
            y=midterm_grade,
            color=sleep_hours,
            color_continuous_scale="Blues",
            title="Study Hours vs Midterm Grade",
            labels={
                study_hours: "Study Hours per Week",
                midterm_grade: "Midterm Grade",
                sleep_hours: "Sleep Hours"
            }
        )
        st.plotly_chart(style_fig(fig_study), use_container_width=True)

    with col4:
        fig_perf_box = px.box(
            data,
            x=midterm_grade,
            y=performance,
            points="all",
            title="Midterm Grades by Self-Reported Performance",
            color_discrete_sequence=[PINK]
        )
        fig_perf_box.update_layout(
            xaxis_title="Midterm Grade",
            yaxis_title=""
        )
        st.plotly_chart(style_fig(fig_perf_box), use_container_width=True)

    with st.expander("View descriptive statistics"):
        st.dataframe(
            data[numeric_columns].describe().T.round(2),
            use_container_width=True
        )


features = [
    age,
    courses,
    study_hours,
    final_grade,
    sleep_hours,
    stress
]

target = midterm_grade

ml_data = data[features + [target]].copy()
ml_data = ml_data.dropna(subset=[target])

for column in features:
    if ml_data[column].notna().any():
        ml_data[column] = ml_data[column].fillna(ml_data[column].median())

usable_features = [
    column for column in features if ml_data[column].notna().any()
]

model_ready = len(ml_data) >= 10 and len(usable_features) > 0

if model_ready:
    X = ml_data[usable_features]
    y = ml_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame({
        "Actual Grade": y_test.values,
        "Predicted Grade": predictions
    })

    results["Residual"] = results["Actual Grade"] - results["Predicted Grade"]
    results["Absolute Error"] = abs(results["Residual"])

    importance = pd.DataFrame({
        "Variable": usable_features,
        "Importance": model.feature_importances_
    })

    importance["Importance (%)"] = importance["Importance"] * 100
    importance = importance.sort_values("Importance (%)", ascending=True)


with tab2:
    st.subheader("Random Forest Regression")

    st.write(
        "The model predicts students' midterm grades using academic, study, sleep, and stress-related variables."
    )

    if not model_ready:
        st.warning(
            "There are not enough valid responses to train and evaluate the model yet."
        )
        st.write(f"Valid responses currently available: {len(ml_data)}")

    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Responses Used", len(ml_data))
        c2.metric("MAE", f"{mae:.2f}")
        c3.metric("RMSE", f"{rmse:.2f}")
        c4.metric("R²", f"{r2:.2f}")

        st.caption(
            "Lower MAE and RMSE indicate smaller prediction errors. "
            "R² closer to 1 indicates stronger explanatory performance."
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            fig_predictions = px.scatter(
                results,
                x="Actual Grade",
                y="Predicted Grade",
                size="Absolute Error",
                color="Absolute Error",
                color_continuous_scale="Turbo",
                title="Actual vs Predicted Grades",
                hover_data=["Absolute Error"]
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
                line=dict(dash="dash", color=SLATE)
            )

            fig_predictions.update_layout(
                xaxis_title="Actual Grade",
                yaxis_title="Predicted Grade"
            )

            st.plotly_chart(
                style_fig(fig_predictions),
                use_container_width=True
            )

        with col2:
            fig_residual = px.scatter(
                results,
                x="Predicted Grade",
                y="Residual",
                color="Absolute Error",
                color_continuous_scale="RdYlBu_r",
                title="Residual Analysis",
                hover_data=["Actual Grade"]
            )

            fig_residual.add_hline(
                y=0,
                line_dash="dash",
                line_color=SLATE
            )

            fig_residual.update_layout(
                xaxis_title="Predicted Grade",
                yaxis_title="Residual (Actual - Predicted)"
            )

            st.plotly_chart(
                style_fig(fig_residual),
                use_container_width=True
            )

        col3, col4 = st.columns(2)

        with col3:
            fig_importance = px.bar(
                importance,
                x="Importance (%)",
                y="Variable",
                orientation="h",
                title="Feature Importance",
                color="Importance (%)",
                color_continuous_scale="Blues"
            )

            fig_importance.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Importance (%)",
                yaxis_title=""
            )

            st.plotly_chart(
                style_fig(fig_importance),
                use_container_width=True
            )

        with col4:
            fig_error = px.histogram(
                results,
                x="Absolute Error",
                nbins=8,
                title="Prediction Error Distribution",
                color_discrete_sequence=[ACCENT]
            )

            fig_error.update_layout(
                xaxis_title="Absolute Error",
                yaxis_title="Number of Predictions",
                showlegend=False
            )

            st.plotly_chart(
                style_fig(fig_error),
                use_container_width=True
            )

        with st.expander("View actual vs predicted values"):
            st.dataframe(
                results.round(2),
                use_container_width=True
            )

        with st.expander("View feature importance table"):
            st.dataframe(
                importance.sort_values(
                    "Importance (%)",
                    ascending=False
                ).round(2),
                use_container_width=True
            )


with tab3:
    st.subheader("Predict a Student's Midterm Grade")

    if not model_ready:
        st.warning(
            "The prediction tool will be available once enough valid responses are collected."
        )

    else:
        st.write(
            "Enter a student's information below to generate a prediction."
        )

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)

            with c1:
                new_age = st.number_input(
                    "Age",
                    min_value=15.0,
                    max_value=80.0,
                    value=float(ml_data[age].median())
                )

                new_courses = st.number_input(
                    "Number of courses",
                    min_value=1.0,
                    value=float(ml_data[courses].median())
                )

                new_study_hours = st.number_input(
                    "Study hours per week",
                    min_value=0.0,
                    value=float(ml_data[study_hours].median())
                )

            with c2:
                new_previous_grade = st.number_input(
                    "Previous grade",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(ml_data[final_grade].median())
                )

                new_sleep = st.number_input(
                    "Sleep hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(ml_data[sleep_hours].median())
                )

                new_stress = st.number_input(
                    "Stress level (1-10)",
                    min_value=1.0,
                    max_value=10.0,
                    value=float(ml_data[stress].median())
                )

            submitted = st.form_submit_button("Generate Prediction")

        if submitted:
            new_student = pd.DataFrame({
                age: [new_age],
                courses: [new_courses],
                study_hours: [new_study_hours],
                final_grade: [new_previous_grade],
                sleep_hours: [new_sleep],
                stress: [new_stress]
            })

            new_student = new_student[usable_features]
            predicted_grade = model.predict(new_student)[0]

            st.success(
                f"Predicted Midterm Grade: {predicted_grade:.2f}"
            )

            st.caption(
                "This prediction is exploratory and depends on the quality "
                "and size of the current survey dataset."
            )
