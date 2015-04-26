#include <Adafruit_NeoPixel.h>

#define PIN 6

// Parameter 1 = number of pixels in strip
// Parameter 2 = pin number (most are valid)
// Parameter 3 = pixel type flags, add together as needed:
//   NEO_KHZ800  800 KHz bitstream (most NeoPixel products w/WS2812 LEDs)
//   NEO_KHZ400  400 KHz (classic 'v1' (not v2) FLORA pixels, WS2811 drivers)
//   NEO_GRB     Pixels are wired for GRB bitstream (most NeoPixel products)
//   NEO_RGB     Pixels are wired for RGB bitstream (v1 FLORA pixels, not v2)
Adafruit_NeoPixel strip = Adafruit_NeoPixel(64, PIN);
    

void setup() {
  strip.begin();
  strip.show();

}

void loop() {
  //strip.setPixelColor(n, red, green, blue);
  strip.setPixelColor(11, 255, 0, 255);
  
  uint32_t magenta = strip.Color(255, 0, 255);
  strip.setPixelColor(13, magenta);
  strip.show();
 
  
}
