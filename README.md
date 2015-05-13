# ez-anti-coordination

There are 3 main files to run. The first is the code to be uploaded to
arduino to run the arduino LED simulation. This can be found in
arduino_side/communication.ino. You need the latest version of the
Arduino software to run this, as well as linking the libraries found in
the libraries folder so the Arduino software knows where to find them.

The second file is anticoordination.py, which contains the main
algorithms and benchmarks. This can be edited to run on its own, by
specifying which benchmark functions to run, or by using the
anticoordination library on its own. Documentation for this is found in
PS3. We added a few extra benchmarks since then, which can be run in the
same way the old benchmarks were run.

The third file is communicate_arduino.py, which when run communicates
over serial with the Arduino, instructing it which LEDs to turn on.
