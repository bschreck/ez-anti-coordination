#include <Adafruit_NeoPixel.h>

#define PIN 6

Adafruit_NeoPixel strip = Adafruit_NeoPixel(64, PIN);
    
uint32_t colors[6];
int num_agents;
int num_channels;

void setup() {
  Serial.begin(9600);
  strip.begin();
  strip.show();
  colors[0] = strip.Color(255,0,0);
  colors[1] = strip.Color(255,255,0);
  colors[2] = strip.Color(0,255,0);
  colors[3] = strip.Color(0,255,255);
  colors[4] = strip.Color(0,0,255);
  colors[5] = strip.Color(255,0,255);
  strip.setBrightness(8);
  
}

void loop() {
  if (Serial.available() > 0){
    int signal = Serial.read();

    if (signal == '9'){
      num_agents = Serial.read()-48;
      num_channels = Serial.read()-48;
      strip.clear();
    }
    else if (signal == 'I'){
      num_agents += 1;
    }
    else if (signal == 'D'){
      num_agents -= 1;
    }
    else{
      signal -= 48;
      int agent_strategies[num_agents];
      for (int i=0; i < 8; i++){
        strip.setPixelColor(signal*8 + i, 0,0,0);
      }
      //read strategies for timestep
      for (int i=0; i < num_agents; i++){
        agent_strategies[i] = (Serial.read() - 48);
        
        int current_led = signal*8 + i;
        if (agent_strategies[i] ==9){ //if agent not trasmitting
          strip.setPixelColor(current_led, 150,150,150); //no LED
        }
        else{
          strip.setPixelColor(current_led, colors[agent_strategies[i]*6/num_channels]);
        }
      }
    }
    strip.show();
  }
 
  
}
