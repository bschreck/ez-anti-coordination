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
        if self.n >= self.c and len(successful_channels) == self.c:
            return 1
        elif self.n < self.c and len(successful_channels) == self.n:
            return 1
        map(lambda c: c.reset(), self.channels)
        return 0

    def run_convergence(self):
        signals_converged = [0]*self.k
        timestep = 0
        while sum(signals_converged) < self.k:
            timestep += 1
            signal = self.signals.value()
            signals_converged[signal] = self.timestep(signal, timestep)
        #print "Converged!"
        return timestep, ["agent %s strategy = %s" % (i, agent.strategy) for i,agent in enumerate(self.agents)]


for i in range(1):
    p = (i+90)/100.
    print p
    total_timesteps = 0
    for j in range(1000):
        sim = Simulator(32, p, 16, 2)
        timesteps, strategies = sim.run_convergence()
        total_timesteps += timesteps
    avg_timesteps = total_timesteps/100.

    print avg_timesteps


#sim = Simulator(32, .3, 16, 2)
#timesteps, strategies = sim.run_convergence()
