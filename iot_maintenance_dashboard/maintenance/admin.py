from django.contrib import admin
from .models import Sensor, SensorData, MaintenanceAlert

@admin.register(Sensor)
class SensorAdmin(admin.ModelAdmin):
    list_display = ('name', 'sensor_type', 'location', 'is_active', 'installation_date')
    list_filter = ('is_active', 'sensor_type', 'location')
    search_fields = ('name', 'description', 'location')

@admin.register(SensorData)
class SensorDataAdmin(admin.ModelAdmin):
    list_display = ('sensor', 'timestamp', 'temperature', 'vibration', 'pressure', 'humidity', 'status')
    list_filter = ('status', 'sensor', 'timestamp')
    date_hierarchy = 'timestamp'

@admin.register(MaintenanceAlert)
class MaintenanceAlertAdmin(admin.ModelAdmin):
    list_display = ('sensor', 'timestamp', 'priority', 'message', 'is_resolved')
    list_filter = ('is_resolved', 'priority', 'sensor')
    search_fields = ('message',)
    date_hierarchy = 'timestamp'
