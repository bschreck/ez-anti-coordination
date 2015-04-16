# import matplotlib.pyplot as plt
import math
import random
class Agent(object):
    def __init__(self, p, k, c):
        self.p = p
        self.strategy = {}
        for signal in range(k):
            self.strategy[signal] = random.randint(0,c-1)
        self.c = c

    def updateStrategy(self, k_t, observation, t):
        #possible heuristic: alter p over time based on t
        #self.p += .001
        channel_busy, channel = observation
        if self.strategy[k_t] > -1:
            if channel_busy:
                #print "channel_busy:", channel_busy
                if random.random() < self.p:
                    self.strategy[k_t] = -1
                else:
                    pass
            else:
                pass
        else:
            if channel_busy:
                pass
            else:
                self.strategy[k_t] = channel

    def set_strategy(self, greedy, n = None):

        if greedy:
            return
        else:
            if (not n):
                return
            for signal,action in self.strategy.items():
                if random.random() > 1.0/n:
                    self.strategy[signal] =  -1


class Signal(object):
    def __init__(self,k, signals = None):
        self.k = k
        if signals:
            self.signals = signals
    def value(self):
        return random.randint(0,self.k-1)


class Channel(object):
    def __init__(self):
        self.count = 0
    def transmit(self):
        self.count += 1
    def reset(self):
        self.count = 0

class Simulator(object):
    #n = number of agents
    #p = probability of strategy
    #c = number of channels
    #k = number of signals
    #optional signals argument passes in specific signals
    #instead of generating them from k
    def __init__(self, n, p, c, k, signals = None):
        self.n = n
        self.c = c
        self.k = k
        self.p = p
        self.agents = [Agent(p, k, c) for agent in range(n)]
        if not signals:
            self.signals = Signal(k)
        else:
            self.signals = Signal(k, signals)
        self.channels = [Channel() for channel in range(c)]

    def num_agents(self):
        return len(self.agents)

    def add_agent(self, agent):
        self.agents.append(agent)
        self.n += 1

    def remove_agent(self, agentIdx):
        self.agents.pop(agentIdx)
        self.n -= 1

    def jain_index(self):
        #print "c: ", self.c
        #print "k: ", self.k
        #print "n: ", self.n
        ck = float(self.c*self.k)
        #print "ck: ", ck
        return ck / (ck + self.n - self.c)

    def timestep(self, k_t, t):
        strategies = [agent.strategy[k_t] for agent in self.agents]

        for strategy in strategies:
            if strategy > -1:
                self.channels[strategy].transmit()

        successful_channels = [i for i,channel in enumerate(self.channels) if channel.count == 1]
        #print "successful_channels:", successful_channels


        empty_channels = [i for i,channel in enumerate(self.channels) if channel.count == 0]

        #print "empty channels:", empty_channels
        for agent in self.agents:
            if agent.strategy[k_t] > -1:
                channel_busy = agent.strategy[k_t] not in successful_channels
                agent.updateStrategy(k_t, (channel_busy, agent.strategy[k_t]),t)
            else:
                new_channel = random.randint(0, self.c-1)
                channel_busy = new_channel not in empty_channels
                agent.updateStrategy(k_t, (channel_busy, new_channel), t)

        #check convergence
        map(lambda c: c.reset(), self.channels)
        if self.n >= self.c and len(successful_channels) == self.c:
            return 1
        elif self.n < self.c and len(successful_channels) == self.n:
            return 1
        return 0

    def run_convergence(self, verbose = False):
        if (verbose):
                print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]
        signals_converged = [0]*self.k
        timestep = 0
        while sum(signals_converged) < self.k:

            timestep += 1
            signal = self.signals.value()
            converged = self.timestep(signal, timestep)
            if signals_converged[signal] == 1 and converged == 0:
                print "signal convergence destroyed"
                signals_converged[signal] = converged
                break
            signals_converged[signal] = converged
            if (verbose):
                print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]
        #print "Converged!"
        return timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]

    def run_num_steps(self, num_steps, verbose = False):
        if verbose:
            print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]

        signals_converged = [0]*self.k
        timestep = 0
       
        while (timestep < num_steps):
            signal = self.signals.value()
            signals_converged[signal] = self.timestep(signal, timestep)
            timestep+=1;
            if (verbose):
                print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]

        return timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]

    def run_growing_population(self, num_agents_final, greedy = True, verbose = False):
        timestep = 0
        while (self.num_agents() <= num_agents_final):
            # We set all signal_converged values to 0, which adds negligible extra time in the polite case
            signals_converged = [0]*self.k
            if (verbose):
                print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]
            while sum(signals_converged) < self.k:

                timestep += 1
                signal = self.signals.value()
                converged = self.timestep(signal, timestep)
                if signals_converged[signal] == 1 and converged == 0:
                    print "signal convergence destroyed"
                    signals_converged[signal] = converged
                    break
                signals_converged[signal] = converged
                if (verbose):
                    print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]
            #print "Converged!"

            if (not self.num_agents() == num_agents_final):
                new_agent = Agent(self.p, self.k, self.c);
                new_agent.set_strategy(greedy, self.num_agents())
                self.add_agent(new_agent)
            else: 
                break
        return timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]

    def run_shrinking_population(self, num_agents_final, greedy = True, verbose = False):
        timestep = 0       
        while (self.num_agents() >= num_agents_final):
            # We set all signal_converged values to 0, which adds negligible extra time in the polite case
            signals_converged = [0]*self.k
            if (verbose):
                print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]
            while sum(signals_converged) < self.k:

                timestep += 1
                signal = self.signals.value()
                converged = self.timestep(signal, timestep)
                if signals_converged[signal] == 1 and converged == 0:
                    print "signal convergence destroyed"
                    signals_converged[signal] = converged
                    break
                signals_converged[signal] = converged
                if (verbose):
                    print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]
            #print "Converged!"

            if (not self.num_agents() == num_agents_final and self.num_agents() >= 1):
                self.remove_agent(random.randint(0,self.num_agents()-1))
            else: 
                break
        return timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]

sim = Simulator(3, .3, 2, 2)
timesteps, strategies = sim.run_shrinking_population(1, False, True)

#for i in range(10):
    #p = (i+90)/100.
    #print p
    #total_timesteps = 0
    #for j in range(100):
        #sim = Simulator(5, p, 5, 5)
        #timesteps, strategies = sim.run_convergence()
        #total_timesteps += timesteps
    #avg_timesteps = total_timesteps/10.

    #print avg_timesteps



# for c in range(63):
#     total_timesteps = 0
#     for j in range(100):
#         sim = Simulator(64, .5, c+1, 64)
#         timesteps, strategies = sim.run_convergence()
#         total_timesteps += timesteps
#     avg_timesteps = total_timesteps/100.
#     print "c = ", c
#     print "timesteps = ",avg_timesteps

#for i in range(62):
    #k = i+2
    #total_timesteps = 0
    #for j in range(100):
        #sim = Simulator(64, .5, 32, k)
        #timesteps, strategies = sim.run_convergence()
        #total_timesteps += timesteps
    #avg_timesteps = total_timesteps/100.
    #print "k = ", k
    #print "timesteps = ",avg_timesteps




#Benchmark 1 (fig. 1 - avg # of steps to convergence for various values of c)
def run_benchmark1(n,p):
    all_avg_timesteps = []
    for c in range(63):
        total_timesteps = 0
        for j in range(100):
            sim = Simulator(n, .5, c+1, n)
            timesteps, strategies = sim.run_convergence()
            total_timesteps += timesteps
        avg_timesteps = total_timesteps/100.
        print "c = ", c
        print "timesteps = ",avg_timesteps
    k = range(1,65)
    print "k:"
    print k
    print "timesteps:"
    print all_avg_timesteps
    plt.plot(k, all_avg_timesteps)
    plt.show()


#Benchmark 2 (fig. 2 = avg # of steps to convergence for various values of k)
#n = 64, c = n/2, p = .5
def run_benchmark2(n, p):
    all_avg_timesteps = []
    for i in range(n-1):
        k = i+2
        total_timesteps = 0
        for j in range(100):
            sim = Simulator(n, p, n/2, k)
            timesteps, strategies = sim.run_convergence()
            total_timesteps += timesteps
        avg_timesteps = total_timesteps/100.
        all_avg_timesteps.append(avg_timesteps)
        print "k = ", k
        print "timesteps = ",avg_timesteps
    k = range(2,65)
    print "k:"
    print k
    print "timesteps:"
    print all_avg_timesteps
    #all_avg_timesteps = [22.16, 36.41, 50.94, 65.46, 84.18, 103.79, 119.11, 138.27, 153.32, 166.81,
            #186.0, 206.77, 223.35, 241.76, 267.49, 292.61, 304.77, 319.51,
            #341.27, 361.38, 379.11, 391.2, 420.47, 430.36, 475.44, 495.3,
            #503.98, 532.87, 546.21, 560.0, 600.15, 609.23, 648.32, 634.12,
            #666.31, 698.28, 699.44, 745.16, 767.57, 784.8, 774.54, 812.04,
            #869.16, 866.34, 870.01, 902.79, 936.05, 936.76, 956.87, 989.47,
            #997.25, 1027.04, 1090.05, 1085.03, 1116.87, 1161.2, 1153.51,
            #1180.88, 1170.32, 1240.14, 1236.57, 1270.25]
    plt.plot(k, all_avg_timesteps)
    plt.show()

#jain_indices_n = []
#jain_indices_n_lg2_n = []
#jain_indices_n_2 = []
#for n in range(5,130):
    #k = n
    #c = 1
    #sim = Simulator(n, .5, c, k)
    #jain_indices_n.append(sim.jain_index())

#for n in range(5,130):
    #k = int(n*math.log(n))
    #c = 1
    #sim = Simulator(n, .5, c, k)
    #jain_indices_n_lg2_n.append(sim.jain_index())

#for n in range(5,130):
    #k = n**2
    #c = 1
    #sim = Simulator(n, .5, c, k)
    #jain_indices_n_2.append(sim.jain_index())
#plt.plot(jain_indices_n)
#plt.plot(jain_indices_n_lg2_n)
#plt.plot(jain_indices_n_2)
#plt.show()

#run_benchmark2(64, 0.5)
# run_benchmark2(64, 0.5)

