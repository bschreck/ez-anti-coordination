import serial
import anticoordination as ac
import time
ser = serial.Serial('/dev/cu.usbmodem1451', 9600, timeout=1)  # open first serial port


def s1():
	#initialize simulation
	n = 4
	p = .5
	c = 2
	k = 2
	sim = ac.Simulator(n, p, c, k)
	results = sim.run_convergence_with_results()

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
		time.sleep(.5)


def s2():
	#initialize simulation
	n = 8
	p = .5
	c = 6
	k = 8
	sim = ac.Simulator(n, p, c, k)
	results = sim.run_convergence_with_results()

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
		time.sleep(.5)

def s3():
	#initialize simulation
	n = 6
	p = .5
	c = 3
	k = 4
	sim = ac.Simulator(n, p, c, k)
	results = sim.run_shrinking_population_with_results(3);

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
		time.sleep(.5)

def s4():
	#initialize simulation
	n = 4
	p = .5
	c = 3
	k = 4
	sig_noise = .01
	chan_noise = .01
	sim = ac.Simulator(n, p, c, k, signal_noise = sig_noise, channel_noise = chan_noise)
	results = sim.run_num_steps_with_results(100)

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
		time.sleep(.5)

def s5():
	#initialize simulation
	n = 8
	p = .5
	c = 2
	k = 2
	sim = ac.Simulator(n, p, c, k)
	results = sim.run_convergence_with_results()

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
		time.sleep(.5)





# s1()

# s2()

# s3()

# Skip me
# s4()

s5()

ser.close()             # close port

