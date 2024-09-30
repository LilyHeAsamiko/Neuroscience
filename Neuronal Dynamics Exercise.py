# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:25:54 2024

@author: Admin
"""
#1. Leaky-integrate-and-fire model
'''
Neurons have intricate morphologies: the central part of the cell is the soma, which contains the genetic information and a large fraction of the molecular machinery. At the soma
'''
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
from neurodynex3.leaky_integrate_and_fire import LIF
from neurodynex3.tools import input_factory, plot_tools

LIF.getting_started()
LIF.print_default_parameters()
#For the default neuron parameters (see above), compute the minimal amplitude Imin
# of a step current to elicitate a spike. You can access the default values in your code and do the calculation with correct units
V_REST = -70*b2.mV
V_RESET = -65*b2.mV
FIRING_THRESHOLD = -50*b2.mV
MEMBRANE_RESISTANCE = 10. * b2.Mohm
MEMBRANE_TIME_SCALE = 8. * b2.ms
ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms
from neurodynex3.leaky_integrate_and_fire import LIF
print("resting potential: {}".format(LIF.V_REST))
#-70. mV

import brian2 as b2
from neurodynex3.leaky_integrate_and_fire import LIF
from neurodynex3.tools import input_factory

# create a step current with amplitude = I_min
step_current = input_factory.get_step_current(
    t_start=5, t_end=100, unit_time=b2.ms,
    amplitude=-70*b2.uamp)  # set I_min to your value

# run the LIF model.
# Note: As we do not specify any model parameters, the simulation runs with the default values
(state_monitor,spike_monitor) = LIF.simulate_LIF_neuron(input_current=step_current, simulation_time = 100 * b2.ms)

# plot I and vm
plot_tools.plot_voltage_and_current_traces(
state_monitor, step_current, title="min input", firing_threshold=LIF.FIRING_THRESHOLD)
print("nr of spikes: {}".format(spike_monitor.count[0]))  # should be 0

#Sketch the f-I curve you expect to see.
#What is the maximum rate at which this neuron can fire?
#Inject currents of different amplitudes (from 0nA to 100nA) into a LIF neuron. For each current, run the simulation for 500ms and determine the firing frequency in Hz. Then plot the f-I curve. Pay attention to the low input current.
import numpy as np
for amp in np.linspace(0,100,10):
    step_current = input_factory.get_step_current(
        t_start=5, t_end=100, unit_time=b2.ms,
        amplitude=amp*b2.uamp)  # set I_min to your value

    # run the LIF model.
    # Note: As we do not specify any model parameters, the simulation runs with the default values
    (state_monitor,spike_monitor) = LIF.simulate_LIF_neuron(input_current=step_current, simulation_time = 100 * b2.ms)

    # plot I and vm
    plot_tools.plot_voltage_and_current_traces(
    state_monitor, step_current, title="min input", firing_threshold=LIF.FIRING_THRESHOLD)
    print("nr of spikes: {}".format(spike_monitor.count[0]))  # should be 0
    

# get a random parameter. provide a random seed to have a reproducible experiment
random_parameters = LIF.get_random_param_set(random_seed=432)

# define your test current
test_current = input_factory.get_step_current(
    t_start=0, t_end=30, unit_time=b2.ms, amplitude= 1.2 * b2.namp)

# probe the neuron. pass the test current AND the random params to the function
state_monitor, spike_monitor = LIF.simulate_random_neuron(test_current, random_parameters)

# plot
plot_tools.plot_voltage_and_current_traces(state_monitor, test_current, title="experiment")

# print the parameters to the console and compare with your estimates
# LIF.print_obfuscated_parameters(random_parameters)

# note the higher resolution when discretizing the sine wave: we specify unit_time=0.1 * b2.ms
sinusoidal_current = input_factory.get_sinusoidal_current(200, 1000, unit_time=0.1 * b2.ms,
                                            amplitude= 2.5 * b2.namp, frequency=250*b2.Hz,
                                            direct_current=10 * b2.namp)

# run the LIF model. By setting the firing threshold to to a high value, we make sure to stay in the linear (non spiking) regime.
(state_monitor, spike_monitor) = LIF.simulate_LIF_neuron(input_current=sinusoidal_current, simulation_time = 120 * b2.ms, firing_threshold=0*b2.mV)

# plot the membrane voltage
plot_tools.plot_voltage_and_current_traces(state_monitor, sinusoidal_current, title="Sinusoidal input current")
print("nr of spikes: {}".format(spike_monitor.count[0]))

# exercise1.4.2 amplitude
for f in np.linspace(10,1000,10):
    sinusoidal_current = input_factory.get_sinusoidal_current(200, 1000, unit_time=0.1 * b2.ms,
                                                amplitude= 2.5 * b2.namp, frequency=f*b2.Hz,
                                                direct_current=10 * b2.namp)

    # run the LIF model.
    # Note: As we do not specify any model parameters, the simulation runs with the default values
    (state_monitor,spike_monitor) = LIF.simulate_LIF_neuron(input_current=step_current, simulation_time = 100 * b2.ms)

    # plot I and vm
    plot_tools.plot_voltage_and_current_traces(state_monitor, sinusoidal_current, title="Sinusoidal input current")
    print("nr of spikes: {}".format(spike_monitor.count[0]))

# exercise1.4.3 phase_shift
for f in np.linspace(10,1000,10):
    sinusoidal_current = input_factory.get_sinusoidal_current(200, 1000, unit_time=0.1 * b2.ms, 
                                                amplitude= 2.5 * b2.namp, frequency=f*b2.Hz, 
                                                direct_current=10 * b2.namp, phase_offset=0.5)

    # run the LIF model.
    # Note: As we do not specify any model parameters, the simulation runs with the default values
    (state_monitor,spike_monitor) = LIF.simulate_LIF_neuron(input_current=step_current, simulation_time = 100 * b2.ms)

    # plot I and vm
    plot_tools.plot_voltage_and_current_traces(state_monitor, sinusoidal_current, title="Sinusoidal input current")
    print("nr of spikes: {}".format(spike_monitor.count[0]))
    
# The Exponential Integrate-and-Fire model
import brian2 as b2
import matplotlib.pyplot as plt
import neurodynex3.exponential_integrate_fire.exp_IF as exp_IF
from neurodynex3.tools import plot_tools, input_factory

input_current = input_factory.get_step_current(
    t_start=20, t_end=120, unit_time=b2.ms, amplitude=0.8 * b2.namp)

state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(
    I_stim=input_current, simulation_time=200*b2.ms)

plot_tools.plot_voltage_and_current_traces(
    state_monitor, input_current,title="step current",
    firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike)
print("nr of spikes: {}".format(spike_monitor.count[0]))

#2.1. Exercise: rehobase threshold
MEMBRANE_TIME_SCALE_tau = 12.0 * b2.ms
MEMBRANE_RESISTANCE_R = 20.0 * b2.Mohm
V_REST = -65.0 * b2.mV
V_RESET = -60.0 * b2.mV
RHEOBASE_THRESHOLD_v_rh = -55.0 * b2.mV
SHARPNESS_delta_T = 2.0 * b2.mV
FIRING_THRESHOLD_v_spike = -30. * b2.mV
'''Modify the code example given above: Call simulate_exponential_IF_neuron() and set the function parameter v_spike to +10mV (which overrides the default value -30mV). What do you expect to happen? How many spikes will be generated?'''
state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(
    v_spike=10.0*b2.mV,I_stim=input_current, simulation_time=200*b2.ms)

plot_tools.plot_voltage_and_current_traces(
    state_monitor, input_current,title="step current",
    firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike)
print("nr of spikes: {}".format(spike_monitor.count[0]))
#nr of spikes: 7 

'''Compute the minimal amplitude Irh
 of a constant input current such that the neuron will elicit a spike. If you are not sure what and how to compute Irh
, have a look at Figure 5.1 and the textbox “Rheobase threshold and interpretation of parameters” in the book.
'''
Iext = 10*b2.namp
f = -(V_RESET-V_REST)+SHARPNESS_delta_T*np.exp((V_RESET-V_REST)/SHARPNESS_delta_T)
#I = du2dt - f/R
I = (f + MEMBRANE_RESISTANCE_R*Iext)/MEMBRANE_TIME_SCALE_tau
'''
Validate your result: Modify the code given above and inject a current of amplitude Irh
 and 300ms duration into the expIF neuron.
input_current = input_factory.get_step_current(
    t_start=20, t_end=120, unit_time=b2.ms, amplitude=0.8 * b2.namp)
'''
state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(
    v_spike=10.0*b2.mV,I_stim=input_current, simulation_time=300*b2.ms)

plot_tools.plot_voltage_and_current_traces(
    state_monitor, input_current,title="step current",
    firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike)
print("nr of spikes: {}".format(spike_monitor.count[0]))

#2.12. Exercise: strength-duration curve
import brian2 as b2
import matplotlib.pyplot as plt
import neurodynex3.exponential_integrate_fire.exp_IF as exp_IF
from neurodynex3.tools import plot_tools, input_factory
'''
Have a look at the following code: for the values i = 0, 2 and 6 we did not provide the minimal amplitude, but the entries in min_amp[i] are set to 0. Complete the min_amp list.
'''
i0=6  #change i and find the value that goes into min_amp
durations = [1,   2,    5,  10,   20,   50, 100]
min_amp =   [0., 4.42, 10., 1.10, .70, .48, 5.0]
spike_number = []

I_amp = min_amp[i0]*b2.namp

for i in range(7):
    t=durations[i] 
    title_txt = "I_amp={}, t={}".format(I_amp, t*b2.ms)
    
    input_current = input_factory.get_step_current(t_start=10, t_end=10+t-1, unit_time=b2.ms, amplitude=I_amp)
    
    state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(I_stim=input_current, simulation_time=(t+20)*b2.ms)
    
    plot_tools.plot_voltage_and_current_traces(state_monitor, input_current,
                                               title=title_txt, firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike,
                                              legend_location=2)
    spike_number.append(spike_monitor.count[0])
    print("nr of spikes: {}".format(spike_monitor.count[0]))
    #0:0 4.42:0 1.1:0 10:5 5,2:1, :

plt.plot(durations, min_amp)
plt.title("Strength-Duration curve")
plt.xlabel("t [ms]")
plt.ylabel("min amplitude [nAmp]")

#3. AdEx: the Adaptive Exponential Integrate-and-Fire model
import brian2 as b2
from neurodynex3.adex_model import AdEx
from neurodynex3.tools import plot_tools, input_factory

current = input_factory.get_step_current(10, 250, 1. * b2.ms, 65.0 * b2.pA)
state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time=400 * b2.ms)
plot_tools.plot_voltage_and_current_traces(state_monitor, current)
print("nr of spikes: {}".format(spike_monitor.count[0]))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(state_monitor.v[0])
plt.title('u')
plt.figure()
plt.plot(state_monitor.w[0])
plt.title('w')


# AdEx.plot_adex_state(state_monitor)
#Use the terminology of Fig. 6.1 in Chapter 6.1.
#Call the function AdEx.simulate_AdEx_neuron() with different parameters and try to create adapting, bursting and irregular firing patterns.

MEMBRANE_TIME_SCALE_tau_m = 5 * b2.ms
MEMBRANE_RESISTANCE_R = 500*b2.Mohm
V_REST = -70.0 * b2.mV
V_RESET = -51.0 * b2.mV
RHEOBASE_THRESHOLD_v_rh = -50.0 * b2.mV
SHARPNESS_delta_T = 2.0 * b2.mV
ADAPTATION_VOLTAGE_COUPLING_a = 0.5 * b2.nS
ADAPTATION_TIME_CONSTANT_tau_w = 100.0 * b2.ms
SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b = 7.0 * b2.pA

