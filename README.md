#  Propulsion Health Management System 

### What is this app?
This is a real-time, interactive anomaly detection and Remaining Useful Life (RUL) prediction dashboard for aircraft gas turbine engines. 
Unlike static analytical tools, this system acts as a live sensor simulator. It allows engineers to artificially degrade critical engine components (like compressors and turbines) and instantly see how a trained Deep Learning model interprets those degradations to predict imminent engine failure. The simulation parameters are modeled directly after the NASA C-MAPSS (FD001) turbofan degradation dataset.

**Live Demo:** [Propulsion Health Management System App](https://propulsion-lifecycle-analytics-dhvvufujoqwuyywfd3ctyf.streamlit.app/)

### Architecture & Tech Stack

* **Frontend & UI:** Streamlit (Deployed on Streamlit Community Cloud)
* **Machine Learning Model:** TensorFlow / Keras (LSTM/Dense Neural Network architecture)
* **Data Preprocessing:** Scikit-learn (MinMaxScaler via Joblib), Pandas, NumPy
* **Data Visualization:** Plotly Graph Objects (Multi-axis dynamic line charts and pressure gauges)

### The Workflow

When a user interacts with the system, the application executes a linear data pipeline:
1. **State Initialization:** The system loads the pre-trained `rul_model.keras` and `scaler.pkl` into memory, caching them to prevent reload latency.
2. **Baseline Generation:** A 30-cycle baseline of 24 sensor/setting values is generated, representing a perfectly healthy turbofan engine.
3. **Signal Degradation & Noise Injection:** Based on user inputs via UI sliders, the system applies linear degradation trends to specific sensors (LPC Temp, LPT Temp, HPC Pressure, Bypass Ratio). High-variance Gaussian noise is layered on top of these trends to mimic real-world thermodynamic sensor scatter.
4. **Data Normalization:** The 30-cycle synthetic data block is passed through the pre-fitted Scikit-learn scaler to normalize values between 0 and 1.
5. **Prediction Inference:** The scaled sequence is reshaped `(1, 30, 24)` and passed into the Keras model.
6. **Telemetry Rendering:** The model outputs an integer representing the Remaining Useful Life (RUL). The frontend instantly updates the critical health status (Healthy, Warning, Critical) and renders the raw multi-sensor telemetry using Plotly.

### Key Features

* **Instant Inference:** Employs Streamlit caching (`@st.cache_resource`) to hold the heavy TensorFlow model in active memory, allowing for real-time predictions as users drag UI sliders.
* <img width="654" height="796" alt="image" src="https://github.com/user-attachments/assets/2a9e1e0d-6cf7-4d19-8b06-4f27e390173e" />
<img width="1917" height="865" alt="image" src="https://github.com/user-attachments/assets/ff993040-730e-472c-bb41-01a15d420b0f" />


* **Realistic Sensor Physics:** Does not just pass static numbers to a model; it generates a 30-cycle time-series array complete with engineered Gaussian noise to accurately simulate real hardware chatter (especially on the LPT temperature probes).
* **Multi-Sensor Telemetry Visualization:** Uses custom-scaled Plotly charts to overlay fundamentally different units (Temperature, Pressure, and RPMs) onto a single readable telemetry dashboard.
* **Explainable Thresholds:** Provides clear UI guardrails and tooltips defining what constitutes a "Healthy" baseline vs. a "Broken" state for each specific sensor.

### Setup instructions

1. **Clone the Repository & Install Dependencies:**
    ```bash
    git clone [https://github.com/your-username/propulsion-lifecycle-analytics.git](https://github.com/your-username/propulsion-lifecycle-analytics.git)
    cd propulsion-lifecycle-analytics
    ```

2. **Create and Activate a Virtual Environment:**
    *(Note: Python 3.10 or 3.11 is strictly required for TensorFlow compatibility)*
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application:**
    ```bash
    streamlit run app.py
    ```
