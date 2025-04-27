# Predictive Maintenance System for IoT Sensors in Farms

This project implements a complete predictive maintenance system for IoT sensors used in agricultural settings. It combines machine learning techniques with a Django-based dashboard to help farmers predict and prevent sensor failures before they occur.

## Project Overview

This system allows to:
- Monitor the health of IoT sensors across different farm zones
- Predict potential sensor failures
- Visualize sensor data and maintenance predictions
- Plan maintenance schedules based on predictive analytics

## Project Structure

```
project/
│
├── predictive_maintenance/      # Core ML pipeline components 
├── iot_maintenance_dashboard/   # Django web application
├── data/                        # Data storage
├── models/                      # Trained ML models
└── requirements.txt             # Python dependencies
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Synthetic Data (Optional)

If you don't have your own sensor data yet:

```bash
python -m predictive_maintenance.generate_data
```

### 5. Preprocess Data

```bash
python -m predictive_maintenance.preprocess_data
```

### 6. Train the Model

```bash
python -m predictive_maintenance.train_model
```

### 7. Run the Django Dashboard

```bash
cd iot_maintenance_dashboard
python manage.py migrate  # Set up the database
python manage.py runserver
```

Then visit `http://127.0.0.1:8000/` in your browser.

## ML Pipeline Workflow

1. **Data Generation/Collection**: Sensor data collection from IoT devices (temperature, humidity, voltage)
2. **Data Preprocessing**
3. **Model Training** 
4. **Prediction**: Applying models to new data to predict potential failures
5. **Visualization**: Showing results on the dashboard

## Features

- **Data Visualization**
- **Failure Prediction**: ML models that predict potential failures days in advance
- **Maintenance Scheduling**: Dashboard to plan maintenance based on predictions
- **Sensor Management**

## Usage Examples

### Running a Complete ML Pipeline

```bash
# Generate synthetic data
python -m predictive_maintenance.generate_data

# Preprocess the data
python -m predictive_maintenance.preprocess_data

# Train the predictive model
python -m predictive_maintenance.train_model

# Make predictions on new data
python -m predictive_maintenance.predict

# Visualize the results
python -m predictive_maintenance.visualize_model
python -m predictive_maintenance.visualize_preprocessed_data
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

