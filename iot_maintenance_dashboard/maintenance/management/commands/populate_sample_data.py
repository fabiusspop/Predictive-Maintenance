import random
import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from maintenance.models import Sensor, SensorData, MaintenanceAlert

class Command(BaseCommand):
    help = 'Populates the database with sample IoT maintenance data for testing'

    def handle(self, *args, **kwargs):
        self.stdout.write('Creating sample data...')
        
        # clear existing data
        self.stdout.write('Clearing existing data...')
        MaintenanceAlert.objects.all().delete()
        SensorData.objects.all().delete()
        Sensor.objects.all().delete()
        
        # sensors
        self.stdout.write('Creating sensors...')
        sensors = []
        
        sensor_types = ['Temperature', 'Pressure', 'Vibration', 'Humidity', 'Flow']
        locations = ['Building A', 'Building B', 'Factory Floor', 'Server Room', 'Warehouse']
        
        # creating sensors
        for i in range(1, 11): 
            active = random.random() > 0.2  # setting: 80% of sensors active
            installation_date = timezone.now().date() - datetime.timedelta(days=random.randint(30, 365))
            
            sensor = Sensor.objects.create(
                name=f'Sensor-{i:03d}',
                description=f'Sample sensor {i} for monitoring {random.choice(sensor_types).lower()}',
                location=random.choice(locations),
                sensor_type=random.choice(sensor_types),
                is_active=active,
                installation_date=installation_date
            )
            sensors.append(sensor)
            self.stdout.write(f'  Created sensor: {sensor.name}')
        
        # sensor data
        self.stdout.write('Creating sensor data...')
        status_choices = ['normal', 'warning', 'critical']
        status_weights = [0.85, 0.1, 0.05]  # 85% normal, 10% warning, 5% critical
        
        for sensor in sensors:
            # create data if active sensor
            if not sensor.is_active:
                continue
                
            # data points for last 24 hours
            base_temp = random.uniform(20, 25)
            base_vibration = random.uniform(0.5, 2)
            base_pressure = random.uniform(990, 1015)
            base_humidity = random.uniform(40, 60)
            
            for hours_ago in range(24, -1, -1):
                timestamp = timezone.now() - datetime.timedelta(hours=hours_ago)
                
                # randomness
                temp_variation = random.uniform(-2, 2)
                vibration_variation = random.uniform(-0.5, 0.5)
                pressure_variation = random.uniform(-5, 5)
                humidity_variation = random.uniform(-5, 5)
                
                # bigger variation --> anomaly, can be set
                if random.random() > 0.9:  # -> 10% anomaly chance
                    temp_variation *= 3
                    vibration_variation *= 3
                
                temperature = base_temp + temp_variation
                vibration = max(0, base_vibration + vibration_variation)
                pressure = base_pressure + pressure_variation
                humidity = max(0, min(100, base_humidity + humidity_variation))
                
                status = random.choices(status_choices, status_weights)[0]
                
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
            
            self.stdout.write(f'  Created data for sensor: {sensor.name}')
        
        # alerts
        self.stdout.write('Creating maintenance alerts...')
        priority_choices = ['low', 'medium', 'high', 'critical']
        
        alert_messages = [
            "Temperature exceeds normal operating range",
            "Vibration levels are abnormally high",
            "Humidity levels outside recommended range",
            "Pressure readings show potential leak",
            "Sensor connectivity issues detected",
            "Routine maintenance required",
            "Battery level critically low",
            "Calibration error detected",
            "Sensor reading fluctuations detected",
            "Physical inspection recommended"
        ]
        
        # assign random alerts for each sensor
        for sensor in sensors:
            for _ in range(random.randint(0, 3)):
                created_days_ago = random.randint(0, 14)  # last 14 days
                created_at = timezone.now() - datetime.timedelta(days=created_days_ago)
                
                priority = random.choice(priority_choices)
                message = random.choice(alert_messages)
                
                # 60% chance of being resolved if more than 3 days old
                is_resolved = created_days_ago > 3 and random.random() > 0.4
                
                resolved_at = None
                if is_resolved:
                    # solved between 1 hour and 2 days after creation
                    hours_until_resolved = random.randint(1, 48)
                    resolved_at = created_at + datetime.timedelta(hours=hours_until_resolved)
                
                alert = MaintenanceAlert.objects.create(
                    sensor=sensor,
                    timestamp=created_at,
                    message=message,
                    priority=priority,
                    is_resolved=is_resolved,
                    resolved_at=resolved_at
                )
                
                self.stdout.write(f'  Created alert: {alert.priority} for {sensor.name}')
        
        self.stdout.write(self.style.SUCCESS('Successfully created sample data!'))
        self.stdout.write(f'Created {Sensor.objects.count()} sensors')
        self.stdout.write(f'Created {SensorData.objects.count()} sensor readings')
        self.stdout.write(f'Created {MaintenanceAlert.objects.count()} maintenance alerts') 