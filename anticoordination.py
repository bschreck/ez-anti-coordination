import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import math
import random
import numpy as np
class Agent(object):
    #mu is for exponential_p
    def __init__(self, p, k, c, mu =  .5, backoff_strategy = "constant"):
        self.p = p
        self.mu = mu
        self.k = k
        self.strategy = {}
        for signal in range(k):
            self.strategy[signal] = random.randint(0,c-1)
        self.c = c

        if backoff_strategy == "linear":
            self.backoff_strategy = self.linear_p
        elif backoff_strategy == "exponential":
            self.backoff_strategy = self.exponential_p
        #backoff_strategy == "constant"
        else:
            self.backoff_strategy = self.constant_p

    def cardinality(self):
        return float(len([s for s in self.strategy if self.strategy[s] > -1]))

    #backoff strategies:
    def constant_p(self):
        return self.p
    def linear_p(self):
        return self.cardinality()/self.k
    def exponential_p(self):
        return self.mu**(1 - (self.cardinality()/self.k))


    def updateStrategy(self, k_t, observation, t):
        channel_busy, channel = observation
        if self.strategy[k_t] > -1:
            if channel_busy:
                #self.backoff_strategy() gives current probability of backoff
                if random.random() < self.backoff_strategy():
                    self.strategy[k_t] = -1
                else:
                    pass
            else:
                pass
        else:
            if channel_busy:
                pass
            #pick a new channel
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
    def __init__(self,k):
        self.k = k
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
    #mu = exponent base for exponential backoff_strategy
    #backoff_strategy is one of ["constant", "linear", "exponential"]
    def __init__(self, n, p, c, k, mu = .5, backoff_strategy = "constant",signal_noise = 0, channel_noise = 0):
        self.n = n
        self.c = c
        self.k = k
        self.p = p
        self.mu = mu
        self.backoff_strategy = backoff_strategy
        self.agents = [Agent(p, k, c, mu, backoff_strategy) for agent in range(n)]
        self.signals = Signal(k)
        self.signal_noise = signal_noise
        self.channel_noise = channel_noise
        self.channels = [Channel() for channel in range(c)]
        self.results = []


    def num_agents(self):
        return len(self.agents)

    def add_agent(self, agent):
        self.agents.append(agent)
        self.n += 1

    def remove_agent(self, agentIdx):
        self.agents.pop(agentIdx)
        self.n -= 1

    def jain_index_constant(self):
        #runs the constant p version of the jain index
        ck = float(self.c*self.k)
        return ck / (ck + self.n - self.c)

    def jain_index(self):
        #runs the version of jain index based on cardinality, for varying
        # backoff probability strategies

        #to be run after convergence
        numerator = sum([a.cardinality() for a in self.agents])**2
        denomenator = self.n*sum([a.cardinality()**2 for a in self.agents])
        return numerator/denomenator

    def noisy_signal(self,k_t):
        noise = self.signals.value()
        if random.random() < self.signal_noise:
            return noise
        else:
            return k_t

    def noisy_channel(self,channel_busy):
        noise = not channel_busy
        if random.random() < self.channel_noise:
            return noise
        else:
            return channel_busy

    def timestep(self, k_t, t):
        #run a single timestep of the algorithm
        strategies = [agent.strategy[self.noisy_signal(k_t)] for agent in self.agents]

        #carry out each agent's strategy
        for strategy in strategies:
            if strategy > -1:
                self.channels[strategy].transmit()

        #channels for which only one agent transmitted
        successful_channels = [i for i,channel in enumerate(self.channels) if channel.count == 1]


        #channels for which no agents transmitted
        empty_channels = [i for i,channel in enumerate(self.channels) if channel.count == 0]


        #update each agent's strategy
        self.run_updates(k_t,t, successful_channels, empty_channels)

        #reset each channel for next round
        map(lambda c: c.reset(), self.channels)

        #check convergence
        if self.n >= self.c and len(successful_channels) == self.c:
            return 1
        elif self.n < self.c and len(successful_channels) == self.n:
            return 1
        return 0

    def run_updates(self, k_t, t, successful_channels, empty_channels):
        #update each agent's strategy
        for agent in self.agents:
            if agent.strategy[k_t] > -1:
                channel_busy = agent.strategy[k_t] not in successful_channels
                agent.updateStrategy(k_t,(self.noisy_channel(channel_busy),agent.strategy[k_t]),t)
            else:
                new_channel= random.randint(0,self.c-1)
                channel_busy=new_channel not in empty_channels
                agent.updateStrategy(k_t,(self.noisy_channel(channel_busy),new_channel),t)


    def run_convergence(self, verbose = False):
        #this is the main function that runs the simulator until it converges
        if (verbose):
                print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]
        signals_converged = [0]*self.k
        timestep = 0
        while sum(signals_converged) < self.k:

            timestep += 1
            signal = self.signals.value()
            converged = self.timestep(signal, timestep)

            signals_converged[signal] = converged
            if (verbose):
                print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]
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

                signals_converged[signal] = converged
                if (verbose):
                    print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]

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

                signals_converged[signal] = converged
                if (verbose):
                    print timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]

            if (not self.num_agents() == num_agents_final and self.num_agents() >= 1):
                self.remove_agent(random.randint(0,self.num_agents()-1))
            else:
                break
        return timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]

    def run_convergence_with_results(self):
        #this is the main function that runs the simulator until it converges
        self.results = []
        signals_converged = [0]*self.k
        timestep = 0
        while sum(signals_converged) < self.k:

            timestep += 1
            signal = self.signals.value()
            list_timestep = [signal]+[agent.strategy[signal] for agent in self.agents]
            self.results.append(list_timestep)
            converged = self.timestep(signal, timestep)

            signals_converged[signal] = converged
            #appends the output of each timestep
            #each row in results is a timestep
            #each column is an agent with its strategy
        print self.results
        return self.results

    def run_growing_population_with_results(self, num_agents_final, greedy = True):
        self.results = []
        timestep = 0
        while (self.num_agents() <= num_agents_final):
            # We set all signal_converged values to 0, which adds negligible extra time in the polite case
            signals_converged = [0]*self.k
            while sum(signals_converged) < self.k:

                timestep += 1
                signal = self.signals.value()
                list_timestep = [signal]+[agent.strategy[signal] for agent in self.agents]
                self.results.append(list_timestep)
                converged = self.timestep(signal, timestep)

                signals_converged[signal] = converged


            if (not self.num_agents() == num_agents_final):
                new_agent = Agent(self.p, self.k, self.c);
                new_agent.set_strategy(greedy, self.num_agents())
                self.add_agent(new_agent)
                self.results.append(['I'])
            else:
                break
        return self.results

    def run_shrinking_population_with_results(self, num_agents_final, greedy = True):
        self.results = []
        timestep = 0
        while (self.num_agents() >= num_agents_final):
            # We set all signal_converged values to 0, which adds negligible extra time in the polite case
            signals_converged = [0]*self.k
            while sum(signals_converged) < self.k:

                timestep += 1
                signal = self.signals.value()
                list_timestep = [signal]+[agent.strategy[signal] for agent in self.agents]
                self.results.append(list_timestep)
                converged = self.timestep(signal, timestep)

                signals_converged[signal] = converged

            if (not self.num_agents() == num_agents_final and self.num_agents() >= 1):
                self.remove_agent(random.randint(0,self.num_agents()-1))
                self.results.append(['D'])
            else:
                break
        return self.results




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
    k = range(1,64)
    print "k:"
    print k
    print "timesteps:"
    print all_avg_timesteps
    plt.plot(k, all_avg_timesteps)
    plt.show()

def run_benchmark_noise_signal(n,p, sig_noise):
    all_avg_timesteps = []
    for c in range(n-1):
        total_timesteps = 0
        for j in range(100):
            sim = Simulator(n, .5, c+1, n, signal_noise = sig_noise)
            timesteps, strategies = sim.run_convergence()
            total_timesteps += timesteps
        avg_timesteps = total_timesteps/100.
        print "c = ", c
        print "timesteps = ",avg_timesteps
        all_avg_timesteps.append(avg_timesteps)
    k = range(1,n)
    print "k:"
    print k
    print "timesteps:"
    print all_avg_timesteps
    plt.plot(k, all_avg_timesteps)
    plt.show()

def run_benchmark_noise_channel(n,p, chan_noise):
    all_avg_timesteps = []
    for c in range(n-1):
        total_timesteps = 0
        for j in range(100):
            sim = Simulator(n, .5, c+1, n, channel_noise = chan_noise)
            timesteps, strategies = sim.run_convergence()
            total_timesteps += timesteps
        avg_timesteps = total_timesteps/100.
        print "c = ", c
        print "timesteps = ",avg_timesteps
        all_avg_timesteps.append(avg_timesteps)
    k = range(1,n)
    print "k:"
    print k
    print "timesteps:"
    print all_avg_timesteps
    plt.plot(k, all_avg_timesteps)
    plt.show()

def run_benchmark_noise_signal_channel(n,p, sig_noise, chan_noise):
    all_avg_timesteps = []
    for c in range(n-1):
        total_timesteps = 0
        for j in range(100):
            sim = Simulator(n, .5, c+1, n, channel_noise = chan_noise)
            timesteps, strategies = sim.run_convergence()
            total_timesteps += timesteps
        avg_timesteps = total_timesteps/100.
        print "c = ", c
        print "timesteps = ",avg_timesteps
        all_avg_timesteps.append(avg_timesteps)
    k = range(1,n)
    print "k:"
    print k
    print "timesteps:"
    print all_avg_timesteps
    plt.plot(k, all_avg_timesteps)
    plt.show()

run_benchmark_noise_signal_channel(8,.5,0.01,.01)



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
    plt.plot(k, all_avg_timesteps)
    plt.show()


#Benchmark 3: Jain Indices (a measure of fairness)
# for various settings of K and N.
# C = 1
def run_fairness_benchmark1():
    jain_indices_n = []
    jain_indices_n_lg2_n = []
    jain_indices_n_2 = []
    for n in range(10,130,10):
        k = n
        c = 1
        sim = Simulator(n, .5, c, k)
        jain_indices_n.append(sim.jain_index_constant())

    for n in range(10,130,10):
        k = int(n*math.log(n,2))
        c = 1
        sim = Simulator(n, .5, c, k)
        jain_indices_n_lg2_n.append(sim.jain_index_constant())

    for n in range(10,130,10):
        k = n**2
        c = 1
        sim = Simulator(n, .5, c, k)
        jain_indices_n_2.append(sim.jain_index_constant())
    domain = np.array(range(10,130,10))
    plt.scatter(domain, np.array(jain_indices_n), c='r')
    plt.plot(domain, jain_indices_n,c='r', label="K = N")
    plt.scatter(domain, np.array(jain_indices_n_lg2_n), c='k')
    plt.plot(domain, jain_indices_n_lg2_n,c='k', label="K = NlgN")
    plt.scatter(domain, np.array(jain_indices_n_2), c='b')
    plt.plot(domain, jain_indices_n_2,c='b', label="K = N^2")
    plt.legend()
    plt.show()

#Benchmark 4: Jain Indices (a measure of fairness)
# for various settings of K and N.
# C = N/2
def run_fairness_benchmark2():
    jain_indices_2 = []
    jain_indices_2_lg2_n = []
    jain_indices_2_n = []
    for n in range(10,130,10):
        k = 2
        c = round(n/2)
        sim = Simulator(n, .5, c, k)
        jain_indices_2.append(sim.jain_index_constant())

    for n in range(10,130,10):
        k = 2*math.log(n,2)
        c = round(n/2)
        sim = Simulator(n, .5, c, k)
        jain_indices_2_lg2_n.append(sim.jain_index_constant())

    for n in range(10,130,10):
        k = 2*n
        c = round(n/2)
        sim = Simulator(n, .5, c, k)
        jain_indices_2_n.append(sim.jain_index_constant())
    domain = np.array(range(10,130,10))
    plt.scatter(domain, np.array(jain_indices_2), c='r')
    plt.plot(domain, jain_indices_2,c='r', label="K = 2")
    plt.scatter(domain, np.array(jain_indices_2_lg2_n), c='k')
    plt.plot(domain, jain_indices_2_lg2_n,c='k', label="K = 2lgN")
    plt.scatter(domain, np.array(jain_indices_2_n), c='b')
    plt.plot(domain, jain_indices_2_n,c='b', label="K = 2N")
    plt.legend()
    plt.show()



#Noisy signals

#Noisy channel reads/writes

#Improving performance by tuning probability for 8 channels and various signals
