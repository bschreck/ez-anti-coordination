import serial
import anticoordination as ac
ser = serial.Serial('/dev/cu.usbmodem1411', 9600, timeout=1)  # open first serial port


#initialize simulation
n = 3
p = .5
c = 3
k = 2
sim = ac.Simulator(n, p, c, k)
results = sim.run_convergence_with_results()

# send parameters
# params = "-1,"+str(n)
# ser.write(params)
# ser.write("\n")

# for timestep in results:
# 	current_signal = timestep[0]
# 	output = str(current_signal) 
# 	for agent_strategy in timestep[1:]:
# 		if agent_strategy == -1:
# 			output += str(9)
# 		else:
# 			output += str(agent_strategy)
# 	ser.write(output)
# 	print output

ser.write("0000")
ser.write("1191")
ser.write("2000")
ser.write("3009")

ser.close()             # close port