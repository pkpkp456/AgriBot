#include <Servo.h>

// Ultrasonic Sensor Pins
const int trigPin = 12;
const int echoPin = 13;

// Motor Control Pins
const int leftMotorForward = 7;
const int leftMotorBackward = 8;
const int rightMotorForward = 9;
const int rightMotorBackward = 10;

// Enable Pins (PWM capable)
const int enA = 5;  // Left motor enable
const int enB = 6;  // Right motor enable

// Speed levels
const int normalSpeed = 200;
const int turningSpeed = 180;
const int stopSpeed = 0;

// Servo for scanning
const int servoPin = 3;
Servo servo;

long duration;
float distance;
int frontDistance, leftDistance, rightDistance;

void setup() {
  Serial.begin(9600);
  Serial.println("Obstacle Avoidance Robot Initializing...");

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  pinMode(leftMotorForward, OUTPUT);
  pinMode(leftMotorBackward, OUTPUT);
  pinMode(rightMotorForward, OUTPUT);
  pinMode(rightMotorBackward, OUTPUT);

  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);

  servo.attach(servoPin);
  servo.write(90); // Center position
  delay(1000);
}

int getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH, 25000); // 25ms timeout (max ~4.3m)

  if (duration == 0) {
    // Timeout: no echo received
    Serial.println("Distance read timeout!");
    return 999;
  }

  distance = duration * 0.0343 / 2.0;
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  return (int)distance;
}

void moveForward() {
  Serial.println("Moving Forward");
  analogWrite(enA, normalSpeed);
  analogWrite(enB, normalSpeed);
  digitalWrite(leftMotorForward, HIGH);
  digitalWrite(leftMotorBackward, LOW);
  digitalWrite(rightMotorForward, HIGH);
  digitalWrite(rightMotorBackward, LOW);
}

void stopMoving() {
  Serial.println("Stopping");
  analogWrite(enA, stopSpeed);
  analogWrite(enB, stopSpeed);
  digitalWrite(leftMotorForward, LOW);
  digitalWrite(leftMotorBackward, LOW);
  digitalWrite(rightMotorForward, LOW);
  digitalWrite(rightMotorBackward, LOW);
}

void turnLeft() {
  Serial.println("Turning Left");
  analogWrite(enA, turningSpeed);
  analogWrite(enB, turningSpeed);
  digitalWrite(leftMotorForward, LOW);
  digitalWrite(leftMotorBackward, HIGH);
  digitalWrite(rightMotorForward, HIGH);
  digitalWrite(rightMotorBackward, LOW);
  delay(500);
  stopMoving();
}

void turnRight() {
  Serial.println("Turning Right");
  analogWrite(enA, turningSpeed);
  analogWrite(enB, turningSpeed);
  digitalWrite(leftMotorForward, HIGH);
  digitalWrite(leftMotorBackward, LOW);
  digitalWrite(rightMotorForward, LOW);
  digitalWrite(rightMotorBackward, HIGH);
  delay(500);
  stopMoving();
}

void loop() {
  servo.write(90); // Center
  delay(400);
  Serial.println("\nChecking Front...");
  frontDistance = getDistance();

  if (frontDistance > 0 && frontDistance < 50) {
    stopMoving();
    delay(500);

    // 🔴 Send "OBSTACLE" to Raspberry Pi
    Serial.println("OBSTACLE");

    // 🟢 Wait for "SCAN_DONE" from Raspberry Pi
    unsigned long startTime = millis();
    bool scanComplete = false;

    while (millis() - startTime < 10000) { // Wait up to 10s
      if (Serial.available()) {
        String response = Serial.readStringUntil('\n');
        response.trim();
        if (response == "SCAN_DONE") {
          scanComplete = true;
          break;
        }
      }
    }

    if (scanComplete) {
      Serial.println("Scan complete, deciding direction...");

      // Scan left
      Serial.println("Checking Left...");
      servo.write(150);
      delay(500);
      leftDistance = getDistance();

      // Scan right
      Serial.println("Checking Right...");
      servo.write(30);
      delay(500);
      rightDistance = getDistance();

      // Reset servo
      servo.write(90);
      delay(500);

      // Decide based on available space
      if (leftDistance > rightDistance && leftDistance > 30) {
        turnLeft();
      } else if (rightDistance > 30) {
        turnRight();
      } else {
        Serial.println("No clear path, waiting...");
        delay(1000);
      }
    } else {
      Serial.println("No scan response from Raspberry Pi!");
      delay(1000); // fallback
    }

    delay(300);
  } else {
    moveForward();
  }

  delay(100);
}
