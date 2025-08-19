// Arduino code for controlling 3-pin laser module

const int LASER_PIN = 9;  // Connect the laser's S (signal) pin to Arduino pin 9
boolean laserState = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize laser pin as output
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);  // Ensure laser is off at startup
  
  Serial.println("Laser Controller Ready");
}

void loop() {
  // Check for incoming serial data
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    
    if (command == "LON") {
      // Turn laser on
      digitalWrite(LASER_PIN, HIGH);
      laserState = true;
    }
  }
}
