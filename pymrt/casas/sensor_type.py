"""This file offers sensor types definitions that matches activity visualizer
"""


class CASASSensorType:
    """CASAS sensor type class that contains all possible sensor types.
    """
    def __init__(self, name, category=None):
        self.name = name
        if category is None:
            self.category = CASASSensorType.guess_category(name)
        else:
            self.category = category
        self.color = default_sensor_category_colors.get(self.category,
                                                        "#FFD3D3D3")

    @staticmethod
    def guess_category(name):
        """Guess sensor category based on name
        """
        if 'Battery' in name:
            return 'Battery'
        if 'Radio' in name or 'Zigbee' in name:
            return 'Radio'
        if 'Motion' in name:
            return 'Motion'
        if 'Door' in name:
            return 'Door'
        if 'LightSwitch' in name:
            return 'LightSwitch'
        if 'Light' in name:
            return 'Light'
        if 'Temperature' in name or 'Thermostat' in name:
            return 'Temperature'
        if 'Item' in name:
            return 'Item'
        return 'Other'

    @staticmethod
    def get_best_category_for_sensor(sensor_types, priority_category=None):
        """Get best category for the sensor
        """
        # Check if sensor_types is a list of string or a list of sensortype
        # items
        if all(isinstance(item, str) for item in sensor_types):
            sensor_types = [
                default_sensor_types.get(item, default_sensor_types["Other"])
                for item in sensor_types
            ]
        assert(all(isinstance(item, CASASSensorType) for item in sensor_types))
        # For intellisense purposes
        if False:
            sensor_types = [CASASSensorType("Dummy", "Other")]
        # Get list of sensor categories
        sensor_categories = set()
        for sensor_type in sensor_types:
            if sensor_type.category not in sensor_categories:
                sensor_categories.add(sensor_type.category)
        # Check category priority
        if priority_category is not None and \
                priority_category in sensor_categories:
            return priority_category
        # Default priority list:
        default_category_priority = ["Motion", "Item", "Door", "Temperature",
                                     "Light", "LightSwitch", "Battery",
                                     "Radio", "Other"]
        for category in default_category_priority:
            if category in sensor_categories:
                return category
        return "Other"

    @staticmethod
    def get_category_color(category):
        return default_sensor_category_colors.get(category, "#FFD3D3D3")


default_sensor_category_colors = {
    "Motion": "#FF0000",          # Colors.Red
    "Door": "#008000",            # Colors.Green
    "Temperature": "#B8860B",     # Colors.DarkGoldenRod
    "Item": "#8A2BE2",            # Colors.BlueViolet
    "Light": "#FFA500",           # Colors.Orange
    "LightSwitch": "#FF80CC",     # Colors.DarkOrange
    "Radio": "#FFB6C1",           # Colors.LightPink
    "Battery": "#20B2AA",         # Colors.LightSeaGreen
    "Other": "#D3D3D3"            # Colors.LightGray
    }


default_sensor_types = {
    "Kinect":
        CASASSensorType("Kinect", category="Other"),
    "Thermostat-Setpoint":
        CASASSensorType("Thermostat-Setpoint", category="Temperature"),
    "RemCAS-Logic":
        CASASSensorType("RemCAS-Logic", category="Other"),
    "Weather-WindGust":
        CASASSensorType("Weather-WindGust", category="Other"),
    "Control4-BatteryVoltage":
        CASASSensorType("Control4-BatteryVoltage", category="Battery"),
    "ExperimenterSwitch-01":
        CASASSensorType("ExperimenterSwitch-01", category="Other"),
    "Weather-Visibility":
        CASASSensorType("Weather-Visibility", category="Other"),
    "light_v2_30":
        CASASSensorType("light_v2_30", category="Other"),
    "Control4-MotionArea":
        CASASSensorType("Control4-MotionArea", category="Motion"),
    "by":
        CASASSensorType("by", category="Other"),
    "Android_Rotation_Vector":
        CASASSensorType("Android_Rotation_Vector", category="Other"),
    "Weather-WindDirection":
        CASASSensorType("Weather-WindDirection", category="Other"),
    "Thermostat-Temperature":
        CASASSensorType("Thermostat-Temperature", category="Temperature"),
    "Weather-Condition":
        CASASSensorType("Weather-Condition", category="Other"),
    "contactsingle_glassbreakdetector_c4_32":
        CASASSensorType("contactsingle_glassbreakdetector_c4_32",
                        category="Other"),
    "Control4-Radio_error":
        CASASSensorType("Control4-Radio_error", category="Radio"),
    "MotionArea-01":
        CASASSensorType("MotionArea-01", category="Motion"),
    "RemCAS-v01":
        CASASSensorType("RemCAS-v01", category="Other"),
    "Temperature":
        CASASSensorType("Temperature", category="Temperature"),
    "Weather-UV":
        CASASSensorType("Weather-UV", category="Other"),
    "Office":
        CASASSensorType("Office", category="Other"),
    "test":
        CASASSensorType("test", category="Other"),
    "Accel-Gyro":
        CASASSensorType("Accel-Gyro", category="Other"),
    "ShimmerTilt-01":
        CASASSensorType("ShimmerTilt-01", category="Other"),
    "WebButton":
        CASASSensorType("WebButton", category="Other"),
    "Control4-LightLevel":
        CASASSensorType("Control4-LightLevel", category="Light"),
    "Zigbee-NetSecCounter":
        CASASSensorType("Zigbee-NetSecCounter", category="Radio"),
    "PromptButton":
        CASASSensorType("PromptButton", category="Other"),
    "EventGenerator":
        CASASSensorType("EventGenerator", category="Other"),
    "OneMeter-kWh":
        CASASSensorType("OneMeter-kWh", category="Other"),
    "rPi-Relay":
        CASASSensorType("rPi-Relay", category="Other"),
    "Weather-Temperature":
        CASASSensorType("Weather-Temperature", category="Temperature"),
    "Android_Gyroscope":
        CASASSensorType("Android_Gyroscope", category="Other"),
    "Zigbee-Channel":
        CASASSensorType("Zigbee-Channel", category="Radio"),
    "Control4-BatteryPercent":
        CASASSensorType("Control4-BatteryPercent", category="Battery"),
    "Control4-Seat":
        CASASSensorType("Control4-Seat", category="Other"),
    "Weather-Humidity":
        CASASSensorType("Weather-Humidity", category="Other"),
    "Thermostat-Link":
        CASASSensorType("Thermostat-Link", category="Temperature"),
    "Inertial_Sensors":
        CASASSensorType("Inertial_Sensors", category="Other"),
    "OneMeter-watts":
        CASASSensorType("OneMeter-watts", category="Other"),
    "Asterisk_Agent":
        CASASSensorType("Asterisk_Agent", category="Other"),
    "Control4-Temperature":
        CASASSensorType("Control4-Temperature", category="Temperature"),
    "RemCAS-UI":
        CASASSensorType("RemCAS-UI", category="Other"),
    "Control4-Relay":
        CASASSensorType("Control4-Relay", category="Other"),
    "ExperimenterSwitch-02":
        CASASSensorType("ExperimenterSwitch-02", category="Other"),
    "Weather-Event":
        CASASSensorType("Weather-Event", category="Other"),
    "AD-01":
        CASASSensorType("AD-01", category="Other"),
    "KVA":
        CASASSensorType("KVA", category="Other"),
    "RAT":
        CASASSensorType("RAT", category="Other"),
    "Door-01":
        CASASSensorType("Door-01", category="Door"),
    "ReminderFrontend":
        CASASSensorType("ReminderFrontend", category="Other"),
    "BMP085_Temperature":
        CASASSensorType("BMP085_Temperature", category="Temperature"),
    "Control4-Door":
        CASASSensorType("Control4-Door", category="Door"),
    "Reminder":
        CASASSensorType("Reminder", category="Other"),
    "K-30_CO2_Sensor":
        CASASSensorType("K-30_CO2_Sensor", category="Other"),
    "BMP085_Pressure":
        CASASSensorType("BMP085_Pressure", category="Other"),
    "Control4-Radio":
        CASASSensorType("Control4-Radio", category="Radio"),
    "Power":
        CASASSensorType("Power", category="Other"),
    "C4-TimeZone":
        CASASSensorType("C4-TimeZone", category="Other"),
    "cardaccess_wirelesscontact_26":
        CASASSensorType("cardaccess_wirelesscontact_26", category="Other"),
    "Asterisk":
        CASASSensorType("Asterisk", category="Other"),
    "GridEye":
        CASASSensorType("GridEye", category="Other"),
    "PromptAudio":
        CASASSensorType("PromptAudio", category="Other"),
    "Control4-Motion":
        CASASSensorType("Control4-Motion", category="Motion"),
    "HeadTracker":
        CASASSensorType("HeadTracker", category="Other"),
    "Prediction":
        CASASSensorType("Prediction", category="Other"),
    "Android_Magnetic_Field":
        CASASSensorType("Android_Magnetic_Field", category="Other"),
    "contactsingle_doorcontactsensor_c4_31":
        CASASSensorType("contactsingle_doorcontactsensor_c4_31", category="Other"),
    "Control4-Light":
        CASASSensorType("Control4-Light", category="Light"),
    "Motion-01":
        CASASSensorType("Motion-01", category="Motion"),
    "Weather-Wind":
        CASASSensorType("Weather-Wind", category="Other"),
    "button":
        CASASSensorType("button", category="Other"),
    "Zigbee-MacAddr":
        CASASSensorType("Zigbee-MacAddr", category="Radio"),
    "Control4-TV":
        CASASSensorType("Control4-TV", category="Other"),
    "unknown":
        CASASSensorType("unknown", category="Other"),
    "Zigbee-Structure":
        CASASSensorType("Zigbee-Structure", category="Radio"),
    "Control4-Button":
        CASASSensorType("Control4-Button", category="Other"),
    "control":
        CASASSensorType("control", category="Other"),
    "Item-01":
        CASASSensorType("Item-01", category="Item"),
    "Weather-Precipitation":
        CASASSensorType("Weather-Precipitation", category="Other"),
    "Insteon":
        CASASSensorType("Insteon", category="Other"),
    "GeneratedEvent":
        CASASSensorType("GeneratedEvent", category="Other"),
    "Thermostat-Heater":
        CASASSensorType("Thermostat-Heater", category="Temperature"),
    "ShimmerSixAxis-01":
        CASASSensorType("ShimmerSixAxis-01", category="Other"),
    "Control4-Item":
        CASASSensorType("Control4-Item", category="Item"),
    "Weather-Pressure":
        CASASSensorType("Weather-Pressure", category="Other"),
    "Control4-LightSensor":
        CASASSensorType("Control4-LightSensor", category="Light"),
    "Android_Accelerometer":
        CASASSensorType("Android_Accelerometer", category="Other"),
    "Thermostat-Window":
        CASASSensorType("Thermostat-Window", category="Temperature"),
    "contactsingle_motionsensor_23":
        CASASSensorType("contactsingle_motionsensor_23", category="Other"),
    "system":
        CASASSensorType("system", category="Other"),
    "Insteon-Relay":
        CASASSensorType("Insteon-Relay", category="Other"),
    "GeneralMotion":
        CASASSensorType("GeneralMotion", category="Motion"),
    "GeneralDoor":
        CASASSensorType("GeneralDoor", category="Door"),
    "GeneralItem":
        CASASSensorType("GeneralItem", category="Item"),
    "GeneralTemperature":
        CASASSensorType("GeneralTemperature", category="Temperature"),
    "GeneralLight":
        CASASSensorType("GeneralLight", category="Light"),
    "GeneralPower":
        CASASSensorType("GeneralPower", category="Power"),
    "Other":
        CASASSensorType("Other", category="Other")
}
