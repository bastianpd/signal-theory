// GENERATED from ml-pipelines/nlp-transformers/data/intents.csv by train_intent.py --write-js. Edit the CSV, not this file.
export const CORPUS = [
 {
  "label": "temperature_alert",
  "text": "Chamber temperature is rising fast and needs a cooling alert"
 },
 {
  "label": "temperature_alert",
  "text": "The oven temperature exceeded 85 degrees"
 },
 {
  "label": "temperature_alert",
  "text": "Alert me when the room gets too hot"
 },
 {
  "label": "temperature_alert",
  "text": "Temperature sensor reads 41 C and climbing"
 },
 {
  "label": "temperature_alert",
  "text": "The server rack is overheating"
 },
 {
  "label": "temperature_alert",
  "text": "It is getting very warm in the greenhouse"
 },
 {
  "label": "temperature_alert",
  "text": "Heat level is above the safe threshold"
 },
 {
  "label": "temperature_alert",
  "text": "Warn me if the temperature drops below 5 degrees"
 },
 {
  "label": "lighting_alert",
  "text": "Illumination dropped sharply in the lab"
 },
 {
  "label": "lighting_alert",
  "text": "Light level is too low so turn on the lamp"
 },
 {
  "label": "lighting_alert",
  "text": "The room went dark unexpectedly"
 },
 {
  "label": "lighting_alert",
  "text": "Lux reading fell under 50"
 },
 {
  "label": "lighting_alert",
  "text": "Brightness is too high near the window"
 },
 {
  "label": "lighting_alert",
  "text": "Notify me if the lights fail"
 },
 {
  "label": "lighting_alert",
  "text": "Lighting in the chamber is dim"
 },
 {
  "label": "lighting_alert",
  "text": "Sunlight sensor shows sudden darkness"
 },
 {
  "label": "actuator_control",
  "text": "Turn on the cooling fan"
 },
 {
  "label": "actuator_control",
  "text": "Switch off the relay"
 },
 {
  "label": "actuator_control",
  "text": "Activate the water pump for ten minutes"
 },
 {
  "label": "actuator_control",
  "text": "Please start the heater"
 },
 {
  "label": "actuator_control",
  "text": "Shut down the ventilation"
 },
 {
  "label": "actuator_control",
  "text": "Enable the LED indicator"
 },
 {
  "label": "actuator_control",
  "text": "Open the valve now"
 },
 {
  "label": "actuator_control",
  "text": "Stop the motor"
 },
 {
  "label": "status_query",
  "text": "What is the current temperature?"
 },
 {
  "label": "status_query",
  "text": "Show me the latest sensor readings"
 },
 {
  "label": "status_query",
  "text": "Is the relay on right now?"
 },
 {
  "label": "status_query",
  "text": "How bright is the room?"
 },
 {
  "label": "status_query",
  "text": "Report device status"
 },
 {
  "label": "status_query",
  "text": "Give me a summary of today's measurements"
 },
 {
  "label": "status_query",
  "text": "Is the ESP32 online?"
 },
 {
  "label": "status_query",
  "text": "What was the last alert?"
 },
 {
  "label": "anomaly_report",
  "text": "Sensor values look strange and unstable"
 },
 {
  "label": "anomaly_report",
  "text": "Unexpected vibration detected on the motor"
 },
 {
  "label": "anomaly_report",
  "text": "There is a sudden spike in readings"
 },
 {
  "label": "anomaly_report",
  "text": "The signal is noisy and erratic"
 },
 {
  "label": "anomaly_report",
  "text": "Something abnormal is happening with the pump"
 },
 {
  "label": "anomaly_report",
  "text": "Readings flatlined and stopped changing"
 },
 {
  "label": "anomaly_report",
  "text": "Unusual pattern in the sensor data"
 },
 {
  "label": "anomaly_report",
  "text": "Possible sensor failure with values out of range"
 },
 {
  "label": "maintenance",
  "text": "Schedule maintenance for the cooling system"
 },
 {
  "label": "maintenance",
  "text": "Replace the battery next week"
 },
 {
  "label": "maintenance",
  "text": "Calibrate the temperature sensor"
 },
 {
  "label": "maintenance",
  "text": "Clean the dust filter on the fan"
 },
 {
  "label": "maintenance",
  "text": "Firmware update is due"
 },
 {
  "label": "maintenance",
  "text": "Inspect the wiring before winter"
 },
 {
  "label": "maintenance",
  "text": "Service the pump this month"
 },
 {
  "label": "maintenance",
  "text": "Remind me to check the relay contacts"
 }
];
export const INTENTS = [...new Set(CORPUS.map((r) => r.label))];
