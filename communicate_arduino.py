import serial
import anticoordination as ac
ser = serial.Serial('/dev/cu.usbmodem1411', 9600, timeout=1)  # open first serial port


#initialize simulation
n = 3
p = .5
c = 2
k = 1
sim = ac.Simulator(n, p, c, k)
results = sim.run_convergence_with_results()

for timestep in results:
	current_signal = timestep[0]
	output = str(current_signal) + ","
	for agent_strategy in timestep[1:]:
		output += str(agent_strategy) + ","
	ser.write(output)
	print output

ser.close()             # close port