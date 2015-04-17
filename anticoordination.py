import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import math
import random
import numpy as np
class Agent(object):
    #mu is for exponential_p
    def __init__(self, p, k, c, mu, backoff_strategy):
        self.p = p
        self.mu = mu
        self.k = k
        self.strategy = {}
        for signal in range(k):
            self.strategy[signal] = random.randint(0,c-1)
        self.c = c
        if backoff_strategy == "constant":
            self.backoff_strategy = self.constant_p
        elif backoff_strategy == "linear":
            self.backoff_strategy = self.linear_p
        elif backoff_strategy == "exponential":
            self.backoff_strategy = self.exponential_p
        #worst agent last
        else:
            self.backoff_strategy = self.worst_agent_last_p



    def cardinality(self):
        return float(len([s for s in self.strategy if self.strategy[s] > -1]))

    def constant_p(self):
        return self.p
    def linear_p(self):
        print "strategy:"
        print self.strategy
        print "cardinality:"
        print self.cardinality()
        print "p = :", self.cardinality()/self.k
        return self.cardinality()/self.k
    def exponential_p(self):
        return self.mu**(1 - (self.cardinality()/self.k))


    def updateStrategy(self, k_t, observation, t):
        channel_busy, channel = observation
        if self.strategy[k_t] > -1:
            if channel_busy:
                if random.random() < self.backoff_strategy():
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
    def __init__(self, n, p, c, k, signals = None, mu = .5, backoff_strategy = "constant"):
        self.n = n
        self.c = c
        self.k = k
        self.p = p
        self.mu = mu
        self.backoff_strategy = backoff_strategy
        #self.agents = [Agent(p, k, c, mu, backoff_strategy) for agent in range(n)]
        if not signals:
            self.signals = Signal(k)
        else:
            self.signals = Signal(k, signals)
        #self.channels = [Channel() for channel in range(c)]


    def num_agents(self):
        return len(self.agents)

    def add_agent(self, agent):
        self.agents.append(agent)
        self.n += 1

    def remove_agent(self, agentIdx):
        self.agents.pop(agentIdx)
        self.n -= 1

    def jain_index_constant(self):
        ck = float(self.c*self.k)
        return ck / (ck + self.n - self.c)
    def jain_index(self):
        #to be run after convergence
        numerator = sum([a.cardinality() for a in self.agents])**2
        denomenator = self.n*sum([a.cardinality()**2 for a in self.agents])
        return numerator/denomenator

    def timestep(self, k_t, t):
        strategies = [agent.strategy[k_t] for agent in self.agents]

        for strategy in strategies:
            if strategy > -1:
                self.channels[strategy].transmit()

        successful_channels = [i for i,channel in enumerate(self.channels) if channel.count == 1]


        empty_channels = [i for i,channel in enumerate(self.channels) if channel.count == 0]


        self.run_updates(k_t,t, successful_channels, empty_channels)

        #check convergence
        map(lambda c: c.reset(), self.channels)
        if self.n >= self.c and len(successful_channels) == self.c:
            return 1
        elif self.n < self.c and len(successful_channels) == self.n:
            return 1
        return 0

    def run_updates(self, k_t, t, successful_channels, empty_channels):
        for agent in self.agents:
            if agent.strategy[k_t] > -1:
                channel_busy = agent.strategy[k_t] not in successful_channels
                agent.updateStrategy(k_t,(channel_busy,agent.strategy[k_t]),t)
            else:
                new_channel= random.randint(0,self.c-1)
                channel_busy=new_channel not in empty_channels
                agent.updateStrategy(k_t,(channel_busy,new_channel),t)


    def run_convergence(self, verbose = False):
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

#sim = Simulator(3, .3, 2, 2)
#timesteps, strategies = sim.run_shrinking_population(1, False, True)

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
    #print "k:"
    #print k
    #print "timesteps:"
    #print all_avg_timesteps
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

def run_backoff_strategy_benchmark(rounds):
    strategies = ["constant", "linear", "exponential"]
    all_avg_jains_per_strategy = []
    all_avg_timesteps_per_strategy = []
    #x axis = n
    #x = np.array(range(10,140,12))
    x = np.array(range(10,20,5))
    lines = []
    total_timesteps = 0
    for strategy in strategies:
        all_avg_jains = []
        all_avg_timesteps = []
        #for n in range(10, 140, 12):
        for n in range(10,20,5):
            c = n/2
            k = int(round(2*math.log(n,2)))

            sum_jain_indices = 0
            for j in range(rounds):
                sim = Simulator(n, .5, c, k, backoff_strategy=strategy)
                timesteps, strategies = sim.run_convergence()
                jain = sim.jain_index()
                total_timesteps += timesteps
                sum_jain_indices += jain
            avg_jain = sum_jain_indices/float(rounds)
            all_avg_jains.append(avg_jain)
            avg_timesteps = total_timesteps/float(rounds)
            all_avg_timesteps.append(avg_timesteps)

            print "n = ", n
            print "jain = ",avg_jain
            print "timesteps =", avg_timesteps
        all_avg_jains_per_strategy.append(all_avg_jains)
        all_avg_timesteps_per_strategy.append(all_avg_timesteps)
    line_styles = ["r", "b", "k"]
    print all_avg_timesteps_per_strategy
    print all_avg_jains_per_strategy
    labels = ["constant", "linear", "exponential"]

    f, axarr = plt.subplots(2, sharex=True)


    for i in range(3):
        axarr[0].scatter(x, np.array(all_avg_jains_per_strategy[i]), c=line_styles[i])
        lines.append(axarr[0].plot(x, np.array(all_avg_jains_per_strategy[i]),
            line_styles[i],label=labels[i]))
    axarr[0].legend()

    for i in range(3):
        axarr[1].scatter(x, np.array(all_avg_timesteps_per_strategy[i]), c=line_styles[i])
        lines.append(axarr[1].plot(x, np.array(all_avg_timesteps_per_strategy[i]),
            line_styles[i],label=labels[i]))
    axarr[1].legend()
    plt.show()

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

#run_benchmark2(64, 0.5)
run_benchmark1(64, 0.5)

#run_backoff_strategy_benchmark(1)
#run_fairness_benchmark1()
