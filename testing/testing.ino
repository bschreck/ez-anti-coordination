#include <Adafruit_NeoPixel.h>

#define PIN 6

Adafruit_NeoPixel strip = Adafruit_NeoPixel(64, PIN);
    
uint32_t colors[3];
int num_agents;

void setup() {
  Serial.begin(9600);
  strip.begin();
  strip.show();
  colors[0] = strip.Color(150,0,0);
  colors[1] = strip.Color(0,150,0);
  colors[2] = strip.Color(0,0,150);
  strip.setBrightness(64);
}

void loop() {
  if (Serial.available() > 0){
    num_agents=3;
    int signal = Serial.read()-48;
    Serial.println(signal);

    if (signal == -100000){
      num_agents = Serial.parseInt();
      Serial.println(num_agents);

    }
    else{
      int agent_strategies[num_agents];
      
      //read strategies for timestep
      for (int i=0; i < num_agents; i++){
        agent_strategies[i] = (Serial.read() - 48);
        Serial.print(i);
        Serial.print(" ");
        Serial.println(agent_strategies[i]);
        
        int current_led = signal*8 + i;
        if (agent_strategies[i] ==9){ //if agent not trasmitting
          strip.setPixelColor(current_led, 150,150,150); //no LED
        }
        else{
          strip.setPixelColor(current_led, colors[agent_strategies[i]]);
        }
      }
    }
    strip.show();
  }
 
  
}
