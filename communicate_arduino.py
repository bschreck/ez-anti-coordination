import serial
import anticoordination as ac
import time
ser = serial.Serial('/dev/cu.usbmodem1451', 9600, timeout=1)  # open first serial port


#initialize simulation
n = 6
p = .5
c = 2
k = 4
sim = ac.Simulator(n, p, c, k)
# results = sim.run_convergence_with_results()
results = sim.run_shrinking_population_with_results(3, True)

# send parameters
params = "9"+str(n)+str(c)
ser.write(params)


for timestep in results:
	current_signal = timestep[0]
	output = str(current_signal) 
	for agent_strategy in timestep[1:]:
		if agent_strategy == -1:
			output += str(9)
		else:
			output += str(agent_strategy)
	ser.write(output)
	print output
	time.sleep(.2)





ser.close()             # close port