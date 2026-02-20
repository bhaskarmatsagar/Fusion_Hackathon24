import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from xgboost import XGBClassifier


STATUS_MAP: Dict[int, str] = {
    0: "Completed",
    1: "Terminated",
    2: "Withdrawn",
    3: "Suspended",
    4: "Recruiting",
}

STATUS_COLORS: Dict[str, str] = {
    "Completed": "#2ecc71",   # Green
    "Recruiting": "#3498db",  # Blue
    "Terminated": "#e74c3c",  # Red
    "Withdrawn": "#e67e22",   # Orange
    "Suspended": "#f1c40f",   # Yellow
}


def _train_model_from_csv(
    csv_path: str = "AERO-BirdsEye-Data.csv",
    model_path: str = "model1.pkl",
) -> Any:
    """
    Fallback: train XGBoost model from CSV if model1.pkl is missing.

    This uses only the required features:
    Sponsor, Start_Year, Start_Month, Phase, Enrollment, Condition
    and encodes Status explicitly using STATUS_MAP.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found and training data '{csv_path}' "
            "is also missing. Please place either the trained model or the CSV "
            "in the application directory."
        )

    df = pd.read_csv(csv_path)

    required_cols = [
        "index",
        "Sponsor",
        "Start_Year",
        "Start_Month",
        "Phase",
        "Enrollment",
        "Status",
        "Condition",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Training data is missing required column '{col}'. "
                "Please ensure the CSV has all required fields."
            )

    # Filter to statuses covered by STATUS_MAP and encode them 0–4 explicitly
    allowed_status_labels = set(STATUS_MAP.values())
    df = df[df["Status"].isin(allowed_status_labels)].copy()
    if df.empty:
        raise ValueError(
            "No rows in CSV match the allowed Status labels "
            f"{sorted(allowed_status_labels)}; cannot train model."
        )

    status_to_int = {label: code for code, label in STATUS_MAP.items()}
    df["Status"] = df["Status"].map(status_to_int)

    # Basic encoding for categorical feature columns
    for col in ["Sponsor", "Phase", "Condition"]:
        df[col], _ = pd.factorize(df[col])

    feature_cols = [
        "index",
        "Sponsor",
        "Start_Year",
        "Start_Month",
        "Phase",
        "Enrollment",
        "Condition",
    ]
    X = df[feature_cols]
    y = df["Status"]

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    joblib.dump(model, model_path)
    return model


@st.cache_resource(show_spinner=False)
def load_model(
    model_path: str = "model1.pkl",
    csv_path: str = "AERO-BirdsEye-Data.csv",
) -> Any:
    """
    Load and cache the trained model.

    If model1.pkl is missing but the CSV is present, automatically train
    a compatible model from the CSV, save it to model1.pkl, and return it.
    """
    if os.path.exists(model_path):
        return joblib.load(model_path)

    # Try to create the model from CSV as a fallback
    model = _train_model_from_csv(csv_path=csv_path, model_path=model_path)
    return model


def decode_status(encoded_value: int) -> str:
    """Decode numeric status to human‑readable label, never exposing raw codes."""
    if encoded_value not in STATUS_MAP:
        return "Unknown"
    return STATUS_MAP[encoded_value]


def predict_status(
    model: Any,
    index_value: float,
    sponsor: float,
    start_year: int,
    start_month: int,
    phase: int,
    enrollment: int,
    condition: float,
) -> str:
    """
    Run prediction and return decoded status label.

    The model is assumed to expect numeric features in the order:
    [index, Sponsor, Start_Year, Start_Month, Phase, Enrollment, Condition]
    """
    features = np.array(
        [
            [
                index_value,
                sponsor,
                start_year,
                start_month,
                phase,
                enrollment,
                condition,
            ]
        ],
        dtype=float,
    )
    encoded_pred = model.predict(features)[0]
    return decode_status(int(encoded_pred))


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, Any]] = []


def save_history(
    inputs: Dict[str, Any],
    prediction: str,
) -> None:
    """Save prediction event to session history, keeping only last 3 entries."""
    init_session_state()
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Sponsor": inputs.get("Sponsor"),
        "Start_Year": inputs.get("Start_Year"),
        "Start_Month": inputs.get("Start_Month"),
        "Phase": inputs.get("Phase"),
        "Enrollment": inputs.get("Enrollment"),
        "Condition": inputs.get("Condition"),
        "Prediction": prediction,
    }
    st.session_state.history.append(entry)
    # Keep only the last 3 predictions
    st.session_state.history = st.session_state.history[-3:]


def get_history_dataframe() -> pd.DataFrame:
    init_session_state()
    if not st.session_state.history:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "Phase",
                "Enrollment",
                "Prediction",
                "Sponsor",
                "Start_Year",
                "Start_Month",
                "Condition",
            ]
        )
    return pd.DataFrame(st.session_state.history)


def plot_graphs(history_df: pd.DataFrame) -> None:
    """Create and render Plotly visual analytics based on prediction history."""
    if history_df.empty:
        st.warning("Visual analytics will appear here after you make predictions.")
        return

    # Ensure Phase is treated as string for nicer labels
    history_df = history_df.copy()
    history_df["Phase"] = history_df["Phase"].astype(str)

    # Graph 1: Bar chart – Phase vs Status count
    st.subheader("Phase-wise Status Distribution")
    try:
        bar_fig = px.bar(
            history_df,
            x="Phase",
            color="Prediction",
            barmode="group",
            title="Count of Predicted Status by Phase",
            color_discrete_map=STATUS_COLORS,
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to render bar chart: {e}")

    # Graph 2: Pie chart – Status distribution
    st.subheader("Overall Status Distribution")
    try:
        pie_fig = px.pie(
            history_df,
            names="Prediction",
            title="Predicted Status Distribution",
            color="Prediction",
            color_discrete_map=STATUS_COLORS,
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to render pie chart: {e}")

    # Graph 3: Line chart – Phase vs Status trend (using decoded labels)
    st.subheader("Phase vs Status Trend")
    try:
        # For trend, sort by Phase (numeric order) and timestamp as a tiebreaker
        trend_df = history_df.copy()
        trend_df["Phase_num"] = trend_df["Phase"].astype(float)
        trend_df = trend_df.sort_values(["Phase_num", "timestamp"])

        line_fig = px.line(
            trend_df,
            x="Phase_num",
            y="Prediction",
            markers=True,
            title="Phase vs Predicted Status Trend",
        )
        line_fig.update_layout(
            xaxis_title="Phase",
            yaxis_title="Status (decoded labels)",
        )
        # Replace x-axis ticks with Phase labels 1–5 if present
        line_fig.update_xaxes(
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=[f"Phase {i}" for i in range(1, 6)],
        )
        st.plotly_chart(line_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to render phase vs status trend: {e}")


def render_status_card(prediction: str) -> None:
    """Display the prediction in a large colored card."""
    color = STATUS_COLORS.get(prediction, "#7f8c8d")  # default gray
    st.markdown(
        f"""
        <div style="
            padding: 1.5rem;
            border-radius: 0.75rem;
            background: linear-gradient(135deg, {color}33, {color});
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            text-align: center;
        ">
            <h3 style="margin-bottom: 0.5rem;">Predicted Status</h3>
            <h1 style="margin: 0; font-size: 2.5rem;">{prediction}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_inputs(
    sponsor: Optional[float],
    start_year: Optional[int],
    start_month: Optional[int],
    phase_label: str,
    enrollment: Optional[int],
    condition: Optional[float],
) -> Optional[str]:
    """Validate user inputs; return error message if any invalid, else None."""
    if sponsor is None:
        return "Sponsor value is required."
    if start_year is None or start_year <= 0:
        return "Start Year must be a positive integer."
    if start_month is None or not (1 <= start_month <= 12):
        return "Start Month must be between 1 and 12."
    if not phase_label:
        return "Phase selection is required."
    if enrollment is None or enrollment <= 0:
        return "Enrollment must be a positive integer."
    if condition is None:
        return "Condition value is required."
    return None


def phase_label_to_numeric(phase_label: str) -> int:
    """Convert 'Phase 1' style label to its numeric representation."""
    try:
        return int(phase_label.split()[-1])
    except Exception:
        return 0


def main() -> None:
    st.set_page_config(
        page_title="Clinical Trial Status Prediction Dashboard",
        page_icon="🧬",
        layout="wide",
    )

    init_session_state()

    st.title("Clinical Trial Status Prediction Dashboard")
    st.markdown(
        "Use this dashboard to estimate the **clinical trial status** based on key trial characteristics. "
        "All status predictions are displayed using **human‑readable labels** only."
    )

    # Sidebar – Input form
    with st.sidebar:
        st.header("Input Parameters")

        index_value = st.number_input(
            "Index (row identifier)",
            min_value=0.0,
            step=1.0,
            format="%.0f",
            help="Use the encoded index value used during model training, if known. "
            "Otherwise you can leave it at the default.",
        )

        sponsor = st.number_input(
            "Sponsor (numeric encoding)",
            min_value=0.0,
            step=1.0,
            format="%.0f",
        )

        col_year, col_month = st.columns(2)
        with col_year:
            start_year = st.number_input(
                "Start Year",
                min_value=1900,
                max_value=datetime.now().year + 5,
                step=1,
                value=datetime.now().year,
            )
        with col_month:
            month_options = {
                "January (1)": 1,
                "February (2)": 2,
                "March (3)": 3,
                "April (4)": 4,
                "May (5)": 5,
                "June (6)": 6,
                "July (7)": 7,
                "August (8)": 8,
                "September (9)": 9,
                "October (10)": 10,
                "November (11)": 11,
                "December (12)": 12,
            }
            month_label = st.selectbox("Start Month", list(month_options.keys()))
            start_month = month_options[month_label]

        phase_label = st.selectbox(
            "Phase",
            [f"Phase {i}" for i in range(1, 6)],
        )

        enrollment = st.number_input(
            "Enrollment",
            min_value=1,
            step=1,
            value=100,
        )

        condition = st.number_input(
            "Condition (numeric encoding)",
            min_value=0.0,
            step=1.0,
            format="%.0f",
        )

        st.markdown("---")
        predict_clicked = st.button("🔍 Predict Status", use_container_width=True)
        clear_history_clicked = st.button("🧹 Clear History", use_container_width=True)

    if clear_history_clicked:
        st.session_state.history = []
        st.success("Prediction history cleared.")

    prediction_label: Optional[str] = None
    model_error: Optional[str] = None

    if predict_clicked:
        # Input validation
        error_msg = validate_inputs(
            sponsor, start_year, start_month, phase_label, enrollment, condition
        )
        if error_msg:
            st.error(error_msg)
        else:
            # Convert phase label to numeric encoding for model
            phase_num = phase_label_to_numeric(phase_label)

            try:
                with st.spinner("Running prediction using the clinical trial model..."):
                    model = load_model()
                    prediction_label = predict_status(
                        model=model,
                        index_value=float(index_value),
                        sponsor=float(sponsor),
                        start_year=int(start_year),
                        start_month=int(start_month),
                        phase=int(phase_num),
                        enrollment=int(enrollment),
                        condition=float(condition),
                    )

                # Save to history with human-readable labels only
                save_history(
                    inputs={
                "Index": index_value,
                        "Sponsor": sponsor,
                        "Start_Year": start_year,
                        "Start_Month": start_month,
                        "Phase": phase_label,
                        "Enrollment": enrollment,
                        "Condition": condition,
                    },
                    prediction=prediction_label,
                )

                st.success("Prediction completed successfully.")
            except FileNotFoundError as e:
                model_error = str(e)
            except Exception as e:
                model_error = f"An error occurred while making prediction: {e}"

    # Layout main sections
    top_container = st.container()
    with top_container:
        col_main, col_metrics = st.columns([2, 1])

        with col_main:
            st.subheader("Prediction Result")
            if model_error:
                st.error(model_error)
            elif prediction_label:
                render_status_card(prediction_label)
            else:
                st.info("Provide input parameters in the sidebar and click **Predict Status**.")

        with col_metrics:
            history_df = get_history_dataframe()
            total_predictions = len(history_df)
            unique_statuses = history_df["Prediction"].nunique() if not history_df.empty else 0
            st.subheader("Overview")

            st.metric(label="Total Predictions", value=total_predictions)
            st.metric(label="Unique Statuses Predicted", value=unique_statuses)

            if not history_df.empty:
                latest_status = history_df.iloc[-1]["Prediction"]
                st.markdown("**Latest Status**")
                color = STATUS_COLORS.get(latest_status, "#7f8c8d")
                st.markdown(
                    f'<span style="color:{color}; font-weight:600;">{latest_status}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("_No predictions yet_")

    st.markdown("---")

    # Detailed sections
    section_pred, section_visuals = st.columns([1, 1])

    with section_pred:
        st.subheader("Prediction Details")
        if prediction_label:
            # Single prediction table
            details = {
                "Index": index_value,
                "Sponsor": sponsor,
                "Start Year": start_year,
                "Start Month": start_month,
                "Phase": phase_label,
                "Enrollment": enrollment,
                "Condition": condition,
                "Predicted Status": prediction_label,
            }
            details_df = pd.DataFrame(
                {"Feature": list(details.keys()), "Value": list(details.values())}
            )
            st.table(details_df)
        else:
            st.warning("Prediction details will appear here after you run a prediction.")

        st.subheader("Prediction History (Last 3)")
        history_df = get_history_dataframe()
        if history_df.empty:
            st.info("Prediction history will appear here after you run predictions.")
        else:
            display_cols = ["timestamp", "Phase", "Enrollment", "Prediction"]
            st.dataframe(
                history_df[display_cols].rename(
                    columns={
                        "timestamp": "Timestamp",
                        "Phase": "Phase",
                        "Enrollment": "Enrollment",
                        "Prediction": "Prediction",
                    }
                ),
                use_container_width=True,
            )

    with section_visuals:
        st.subheader("Visual Analytics")
        history_df = get_history_dataframe()
        plot_graphs(history_df)

    st.markdown("---")
    st.subheader("Phase Comparison")
    if history_df.empty:
        st.info("Phase comparison insights will be available after you make predictions.")
    else:
        # Simple textual insight using decoded labels only
        phase_group = history_df.groupby("Phase")["Prediction"].agg(
            lambda x: x.value_counts().idxmax()
        )
        for phase, status in phase_group.items():
            color = STATUS_COLORS.get(status, "#7f8c8d")
            st.markdown(
                f"- **{phase}** most frequently predicted as "
                f'<span style="color:{color}; font-weight:600;">{status}</span>',
                unsafe_allow_html=True,
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>"
        "Clinical Trial Status Prediction Dashboard · Built with Streamlit & Plotly · "
        "All statuses are displayed using decoded labels only."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

