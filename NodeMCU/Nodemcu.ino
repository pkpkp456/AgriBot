#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <UniversalTelegramBot.h>
#include <DHT.h>

// Wi-Fi credentials
const char* ssid = "tecno camon";
const char* password = "DHARANI@123";

// Telegram BOT credentials (no semicolons here!)
#define BOT_TOKEN "7962497571:AAFXZyhnZDv36aCFMf_fWJ4vXRiSh4HyJ_s"
#define CHAT_ID "5073577271"

// DHT Sensor setup
#define DHTPIN 4     // GPIO 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// Soil Moisture Sensor
#define SOIL_PIN 34  // GPIO34 (Analog input)

// Telegram setup
WiFiClientSecure client;
UniversalTelegramBot bot(BOT_TOKEN, client);

// Timing control
unsigned long lastTimeBotRan = 0;
const int delayBotInterval = 10000; // 10 seconds

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Starting...");

  dht.begin();

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi connected");

  // Telegram requires HTTPS
  client.setInsecure();  // Disable SSL certificate verification

  // Notify on boot
  bool sent = bot.sendMessage(CHAT_ID, "ESP32 Bot started!", "");
  Serial.println(sent ? "Startup message sent!" : "Failed to send startup message");
}

void loop() {
  if (millis() - lastTimeBotRan > delayBotInterval) {
    lastTimeBotRan = millis();
    Serial.println("\nReading sensors...");

    // Read sensors
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();
    int soilMoistureRaw = analogRead(SOIL_PIN);

    // Check for DHT read errors
    if (isnan(humidity) || isnan(temperature)) {
      Serial.println("Failed to read from DHT sensor!");
      return;
    }

    // Map soil moisture raw reading (calibrate for your sensor)
    int soilPercent = map(soilMoistureRaw, 4095, 0, 0, 100);
    soilPercent = constrain(soilPercent, 0, 100);  // Clamp values

    // Print sensor values
    Serial.printf("Temperature: %.2f °C\n", temperature);
    Serial.printf("Humidity: %.2f %%\n", humidity);
    Serial.printf("Soil Moisture Raw: %d\n", soilMoistureRaw);
   

    // Format message for Telegram
    String message = "🌡️ Temp: " + String(temperature, 2) + " °C\n";
    message += "💧 Humidity: " + String(humidity, 2) + " %\n";
    message += "🌱 Soil Moisture: " + String(soilPercent) + " %";

    // Send message to Telegram
    bool sent = bot.sendMessage(CHAT_ID, message, "");
    Serial.println(sent ? "Message sent to Telegram!" : "Failed to send message!");
  }
  delay(30000);
}
