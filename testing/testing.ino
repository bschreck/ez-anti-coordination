#include <Adafruit_NeoPixel.h>
#include <Console.h>

#define PIN 6

Adafruit_NeoPixel strip = Adafruit_NeoPixel(64, PIN);
    
uint32_t colors[3];

void setup() {
  strip.begin();
  strip.show();
  colors[0] = strip.Color(150,0,0);
  colors[1] = strip.Color(0,150,0);
  colors[2] = strip.Color(0,0,150);
  strip.setBrightness(64);
  Console.begin();

}

void loop() {
  //strip.setPixelColor(n, red, green, blue);
//  strip.setPixelColor(11, 255, 0, 255);
//  
//  uint32_t magenta = strip.Color(255, 0, 255);
//  strip.setPixelColor(13, magenta);
//  strip.show();
  
  int num_agents = 3;
  int agent_strategies[num_agents];
  if (Serial.available() > 0){
    int signal = Serial.parseInt();
    Console.print(signal);
    Serial.read(); //skip comma
    for (int i=0; i < num_agents; i++){
      agent_strategies[i] = Serial.parseInt();
      Console.print(" ");
      Console.print(agent_strategies[i]);
      if (agent_strategies[i] ==-1){
        strip.setPixelColor(i, 150,150,150);
      }
      else{
        strip.setPixelColor(i, colors[agent_strategies[i]]);
      }
      Serial.read(); //skip comma
    }
    strip.show();
    Console.println();
    delay(1000);
  }
 
  
}
