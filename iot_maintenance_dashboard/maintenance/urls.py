from django.urls import path
from . import views

app_name = 'maintenance'

urlpatterns = [
    # URL pattern for the dashboard home view (to be implemented)
    path('', views.dashboard_home, name='dashboard_home'),
    
    # URL patterns for sensors
    path('sensors/', views.sensor_list, name='sensor_list'),
    path('sensors/<int:pk>/', views.sensor_detail, name='sensor_detail'),
    
    # URL patterns for sensor data
    path('data/', views.sensor_data_list, name='sensor_data_list'),
    
    # URL patterns for maintenance alerts
    path('alerts/', views.alert_list, name='alert_list'),
    path('alerts/<int:pk>/', views.alert_detail, name='alert_detail'),
] 