# Predictive Maintenance for IoT Sensors in Farms

A simple Django application for predictive maintenance of IoT sensors in agricultural settings.

## Project Structure
- `data_generation/`: Scripts for generating synthetic sensor data
- `data_preprocessing/`: Data cleaning and feature engineering
- `training/`: ML model training
- `prediction/`: Failure prediction based on sensor data
- `web/`: Django web interface

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run Django server:
   ```
   cd web
   python manage.py runserver
   ```

## Modules
- Data Generation: Creates synthetic data for 3 sensor types
- Data Preprocessing: Cleans and prepares data for ML
- Training: Builds and trains predictive models
- Prediction: Applies models to detect potential failures 