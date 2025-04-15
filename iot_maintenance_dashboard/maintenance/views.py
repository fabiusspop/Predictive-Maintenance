from django.shortcuts import render, get_object_or_404
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
