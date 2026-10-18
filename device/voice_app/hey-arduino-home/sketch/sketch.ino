/*
 * ====================================================================================================
 * Sign & Say - Smart Home Mobility Project
 * Author: Umit D. Sami
 * Open source project - ExecuTorch Hackathon 2026 (San Francisco) - 11.18.2026
 *
 * This is a smart home mobility project for physically challenged individuals. The project is an IoT
 * system running on the Arduino VENTUNO Q without the use of any cloud resources (everything runs inside
 * the device, including advanced AI computation tasks such as speech and vision recognition).
 * The project allows a physically challenged person to set the temperature in the house, turn the lights
 * on and off, and change TV channels in two ways:
 *   1. VENTUNO Q Voice AI assistant module
 *   2. VENTUNO Q Hand Recognition with ExecuTorch running in the device
 * ====================================================================================================
 */
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <math.h>

ArduinoLEDMatrix matrix;

// 104 LEDs (13 columns x 8 rows), 3-bit grayscale
uint8_t logicalFrame[8 * 13];

// 0: IDLE, 1: LISTENING, 2: PROCESSING, 3: COMMAND DONE, 4: NOT UNDERSTOOD
int currentState = 0;
float scanPos = 0.0;
float scanDir = 1.0;

const uint8_t CHECK[][2] = {{4, 2}, {5, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7}, {2, 8}, {1, 9}, {0, 10}};
const uint8_t CROSS[][2] = {{1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}, {6, 8},
                            {1, 8}, {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3}};

void setSystemState(int state) {
  currentState = constrain(state, 0, 4);
  if (currentState == 1) {
    scanPos = 0.0;
  }
}

uint8_t scanBrightness(int col, float headPos, float tailLen) {
  float dist = fabs((float)col - headPos);
  if (dist > tailLen) return 0;
  float t = 1.0 - (dist / tailLen);
  return (uint8_t)(t * t * t * 7.0 + 0.5);
}

void drawScanner() {
  float scanSpeed = (currentState == 1) ? 0.6 : 0.2;
  scanPos += scanDir * scanSpeed;
  if (scanPos >= 12.0) { scanPos = 12.0; scanDir = -1.0; }
  if (scanPos <= 0.0 && scanDir < 0) { scanPos = 0.0; scanDir = 1.0; }

  for (int col = 0; col < 13; col++) {
    logicalFrame[3 * 13 + col] = 1;
    logicalFrame[4 * 13 + col] = 1;
    uint8_t bri = scanBrightness(col, scanPos, 4.0);
    if (bri > 0) {
      logicalFrame[3 * 13 + col] = max((int)logicalFrame[3 * 13 + col], (int)bri);
      logicalFrame[4 * 13 + col] = max((int)logicalFrame[4 * 13 + col], (int)bri);
    }
  }
}

template <size_t N>
void drawPoints(const uint8_t (&points)[N][2]) {
  for (size_t i = 0; i < N; i++) {
    logicalFrame[points[i][0] * 13 + points[i][1]] = 7;
  }
}

void renderMatrix() {
  memset(logicalFrame, 0, sizeof(logicalFrame));
  if (currentState == 1 || currentState == 2) drawScanner();
  if (currentState == 3) drawPoints(CHECK);
  if (currentState == 4) drawPoints(CROSS);
  matrix.draw(logicalFrame);
}

void setup() {
  delay(1000);
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();

  Bridge.begin();
  Bridge.provide("set_state", setSystemState);
}

void loop() {
  if (currentState != 0) {
    renderMatrix();
  } else {
    matrix.clear();
  }
  delay(30);
}
