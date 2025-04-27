from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.utils import timezone
import random
import datetime
from .models import Sensor, SensorData, MaintenanceAlert

# Create your views here.

def dashboard_home(request):
    """View for the main dashboard page"""
    sensors = Sensor.objects.all()
    alerts = MaintenanceAlert.objects.filter(is_resolved=False).order_by('-priority', '-timestamp')[:5]
    recent_data = SensorData.objects.all().order_by('-timestamp')[:10]
    
    context = {
        'sensors': sensors,
        'alerts': alerts,
        'recent_data': recent_data,
        'active_sensors': sensors.filter(is_active=True).count(),
        'total_sensors': sensors.count(),
        'total_alerts': MaintenanceAlert.objects.filter(is_resolved=False).count(),
    }
    return render(request, 'maintenance/dashboard_home.html', context)

def sensor_list(request):
    """View for listing all sensors"""
    sensors = Sensor.objects.all()
    context = {'sensors': sensors}
    return render(request, 'maintenance/sensor_list.html', context)

def sensor_detail(request, pk):
    """View for detailed information about a specific sensor"""
    sensor = get_object_or_404(Sensor, pk=pk)
    readings = sensor.readings.order_by('-timestamp')[:50]  # Get last 50 readings
    alerts = sensor.alerts.order_by('-timestamp')[:10]  # Get last 10 alerts
    
    context = {
        'sensor': sensor,
        'readings': readings,
        'alerts': alerts,
    }
    return render(request, 'maintenance/sensor_detail.html', context)

def sensor_data_list(request):
    """View for listing sensor data/readings"""
    data = SensorData.objects.all().order_by('-timestamp')[:100]  # Get last 100 readings
    context = {'data': data}
    return render(request, 'maintenance/sensor_data_list.html', context)

def alert_list(request):
    """View for listing all maintenance alerts"""
    alerts = MaintenanceAlert.objects.all().order_by('-timestamp')
    
    # Calculate counts for different priorities
    critical_count = MaintenanceAlert.objects.filter(priority='critical', is_resolved=False).count()
    high_count = MaintenanceAlert.objects.filter(priority='high', is_resolved=False).count()
    medium_count = MaintenanceAlert.objects.filter(priority='medium', is_resolved=False).count()
    low_count = MaintenanceAlert.objects.filter(priority='low', is_resolved=False).count()
    
    context = {
        'alerts': alerts,
        'critical_count': critical_count,
        'high_count': high_count,
        'medium_count': medium_count,
        'low_count': low_count,
    }
    return render(request, 'maintenance/alert_list.html', context)

def alert_detail(request, pk):
    """View for detailed information about a specific alert"""
    alert = get_object_or_404(MaintenanceAlert, pk=pk)
    context = {'alert': alert}
    return render(request, 'maintenance/alert_detail.html', context)

# Utility Functions

def generate_data(request):
    """View for generating sample data"""
    # Get active sensors
    active_sensors = Sensor.objects.filter(is_active=True)
    
    if not active_sensors.exists():
        messages.error(request, "No active sensors found. Please activate sensors first.")
        return redirect('maintenance:dashboard_home')
    
    # Generate data for each active sensor
    data_count = 0
    status_choices = ['normal', 'warning', 'critical']
    status_weights = [0.85, 0.1, 0.05]
    
    for sensor in active_sensors:
        # Create data for the last hour
        base_temp = random.uniform(20, 25)
        base_vibration = random.uniform(0.5, 2)
        base_pressure = random.uniform(990, 1015)
        base_humidity = random.uniform(40, 60)
        
        # Generate 10 readings per sensor
        for minutes_ago in range(60, 0, -6):  # Every 6 minutes, for the last hour
            timestamp = timezone.now() - datetime.timedelta(minutes=minutes_ago)
            
            # Add some randomness to the values
            temp_variation = random.uniform(-1, 1)
            vibration_variation = random.uniform(-0.2, 0.2)
            pressure_variation = random.uniform(-2, 2)
            humidity_variation = random.uniform(-3, 3)
            
            # If we want an anomaly, make a bigger variation
            if random.random() > 0.9:  # 10% chance of anomaly
                temp_variation *= 3
                vibration_variation *= 3
            
            temperature = base_temp + temp_variation
            vibration = max(0, base_vibration + vibration_variation)
            pressure = base_pressure + pressure_variation
            humidity = max(0, min(100, base_humidity + humidity_variation))
            
            # Determine status based on the values
            status = random.choices(status_choices, status_weights)[0]
            
            # Some sensors may not have all types of readings
            if sensor.sensor_type != 'Temperature':
                temperature = None
            if sensor.sensor_type != 'Vibration':
                vibration = None
            if sensor.sensor_type != 'Pressure':
                pressure = None
            if sensor.sensor_type != 'Humidity':
                humidity = None
            
            SensorData.objects.create(
                sensor=sensor,
                timestamp=timestamp,
                temperature=temperature,
                vibration=vibration,
                pressure=pressure,
                humidity=humidity,
                status=status
            )
            data_count += 1
    
    # Generate random alerts based on the new data
    alert_count = 0
    for sensor in active_sensors:
        # 30% chance to generate an alert per sensor
        if random.random() < 0.3:
            priority_choices = ['low', 'medium', 'high', 'critical']
            priority_weights = [0.4, 0.3, 0.2, 0.1]  # Weighted towards less severe
            
            alert_messages = [
                "Temperature exceeds normal operating range",
                "Vibration levels are abnormally high",
                "Humidity levels outside recommended range",
                "Pressure readings show potential leak",
                "Sensor connectivity issues detected",
                "Routine maintenance required"
            ]
            
            MaintenanceAlert.objects.create(
                sensor=sensor,
                timestamp=timezone.now() - datetime.timedelta(minutes=random.randint(0, 60)),
                message=random.choice(alert_messages),
                priority=random.choices(priority_choices, priority_weights)[0],
                is_resolved=False
            )
            alert_count += 1
    
    # Success message
    messages.success(request, f"Successfully generated {data_count} new data points and {alert_count} alerts.")
    return redirect('maintenance:dashboard_home')

def train_model(request):
    """View for simulating model training"""
    # Just a placeholder for now
    import time
    time.sleep(2)  # Simulate training time
    
    messages.success(request, "Model training completed successfully! Predictive maintenance model is now up to date.")
    return redirect('maintenance:dashboard_home')

def predict(request):
    """View for simulating predictions"""
    # Get active sensors
    active_sensors = Sensor.objects.filter(is_active=True)
    
    if not active_sensors.exists():
        messages.error(request, "No active sensors found. Cannot make predictions.")
        return redirect('maintenance:dashboard_home')
    
    # Randomly select some sensors to generate predictive alerts for
    prediction_count = 0
    for sensor in active_sensors:
        # 20% chance to predict an issue for each sensor
        if random.random() < 0.2:
            prediction_messages = [
                "Predicted failure within 24 hours based on temperature pattern",
                "Maintenance recommended - vibration patterns indicate bearing wear",
                "Humidity levels trending toward critical threshold",
                "Predictive model suggests calibration needed within 72 hours",
                "Deteriorating performance detected - maintenance advised"
            ]
            
            # Create a predictive alert
            MaintenanceAlert.objects.create(
                sensor=sensor,
                timestamp=timezone.now(),
                message=f"PREDICTION: {random.choice(prediction_messages)}",
                priority='medium',  # Most predictions are medium priority
                is_resolved=False
            )
            prediction_count += 1
    
    if prediction_count > 0:
        messages.success(request, f"Prediction completed. Generated {prediction_count} predictive maintenance alerts.")
    else:
        messages.info(request, "Prediction completed. No maintenance issues predicted at this time.")
    
    return redirect('maintenance:dashboard_home')
