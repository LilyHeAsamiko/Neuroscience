# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:25:54 2024

@author: Admin
"""
# 1. Leaky-integrate-and-fire model
'''
Neurons have intricate morphologies: the central part of the cell is the soma, which contains the genetic information and a large fraction of the molecular machinery. At the soma
'''

import neurodynex3.hopfield_network.plot_tools as hfplot
import numpy
from neurodynex3.hopfield_network import network, pattern_tools, plot_tools
from neurodynex3.adex_model import AdEx
from neurodynex3.tools import plot_tools, input_factory
import neurodynex3.exponential_integrate_fire.exp_IF as exp_IF
from neurodynex3.tools import input_factory
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
from neurodynex3.leaky_integrate_and_fire import LIF
from neurodynex3.tools import input_factory, plot_tools
LIF.getting_started()
LIF.print_default_parameters()
# For the default neuron parameters (see above), compute the minimal amplitude Imin
# of a step current to elicitate a spike. You can access the default values in your code and do the calculation with correct units
V_REST = -70*b2.mV
V_RESET = -65*b2.mV
FIRING_THRESHOLD = -50*b2.mV
MEMBRANE_RESISTANCE = 10. * b2.Mohm
MEMBRANE_TIME_SCALE = 8. * b2.ms
ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms
print("resting potential: {}".format(LIF.V_REST))
# -70. mV


# create a step current with amplitude = I_min
step_current = input_factory.get_step_current(
    t_start=5, t_end=100, unit_time=b2.ms,
    amplitude=-70*b2.uamp)  # set I_min to your value

# run the LIF model.
# Note: As we do not specify any model parameters, the simulation runs with the default values
(state_monitor, spike_monitor) = LIF.simulate_LIF_neuron(
    input_current=step_current, simulation_time=100 * b2.ms)

# plot I and vm
plot_tools.plot_voltage_and_current_traces(
    state_monitor, step_current, title="min input", firing_threshold=LIF.FIRING_THRESHOLD)
print("nr of spikes: {}".format(spike_monitor.count[0]))  # should be 0

# Sketch the f-I curve you expect to see.
# What is the maximum rate at which this neuron can fire?
# Inject currents of different amplitudes (from 0nA to 100nA) into a LIF neuron. For each current, run the simulation for 500ms and determine the firing frequency in Hz. Then plot the f-I curve. Pay attention to the low input current.
for amp in np.linspace(0, 100, 10):
    step_current = input_factory.get_step_current(
        t_start=5, t_end=100, unit_time=b2.ms,
        amplitude=amp*b2.uamp)  # set I_min to your value

    # run the LIF model.
    # Note: As we do not specify any model parameters, the simulation runs with the default values
    (state_monitor, spike_monitor) = LIF.simulate_LIF_neuron(
        input_current=step_current, simulation_time=100 * b2.ms)

    # plot I and vm
    plot_tools.plot_voltage_and_current_traces(
        state_monitor, step_current, title="min input", firing_threshold=LIF.FIRING_THRESHOLD)
    print("nr of spikes: {}".format(spike_monitor.count[0]))  # should be 0


# get a random parameter. provide a random seed to have a reproducible experiment
random_parameters = LIF.get_random_param_set(random_seed=432)

# define your test current
test_current = input_factory.get_step_current(
    t_start=0, t_end=30, unit_time=b2.ms, amplitude=1.2 * b2.namp)

# probe the neuron. pass the test current AND the random params to the function
state_monitor, spike_monitor = LIF.simulate_random_neuron(
    test_current, random_parameters)

# plot
plot_tools.plot_voltage_and_current_traces(
    state_monitor, test_current, title="experiment")

# print the parameters to the console and compare with your estimates
# LIF.print_obfuscated_parameters(random_parameters)

# note the higher resolution when discretizing the sine wave: we specify unit_time=0.1 * b2.ms
sinusoidal_current = input_factory.get_sinusoidal_current(200, 1000, unit_time=0.1 * b2.ms,
                                                          amplitude=2.5 * b2.namp, frequency=250*b2.Hz,
                                                          direct_current=10 * b2.namp)

# run the LIF model. By setting the firing threshold to to a high value, we make sure to stay in the linear (non spiking) regime.
(state_monitor, spike_monitor) = LIF.simulate_LIF_neuron(
    input_current=sinusoidal_current, simulation_time=120 * b2.ms, firing_threshold=0*b2.mV)

# plot the membrane voltage
plot_tools.plot_voltage_and_current_traces(
    state_monitor, sinusoidal_current, title="Sinusoidal input current")
print("nr of spikes: {}".format(spike_monitor.count[0]))

# exercise1.4.2 amplitude
for f in np.linspace(10, 1000, 10):
    sinusoidal_current = input_factory.get_sinusoidal_current(200, 1000, unit_time=0.1 * b2.ms,
                                                              amplitude=2.5 * b2.namp, frequency=f*b2.Hz,
                                                              direct_current=10 * b2.namp)

    # run the LIF model.
    # Note: As we do not specify any model parameters, the simulation runs with the default values
    (state_monitor, spike_monitor) = LIF.simulate_LIF_neuron(
        input_current=step_current, simulation_time=100 * b2.ms)

    # plot I and vm
    plot_tools.plot_voltage_and_current_traces(
        state_monitor, sinusoidal_current, title="Sinusoidal input current")
    print("nr of spikes: {}".format(spike_monitor.count[0]))

# exercise1.4.3 phase_shift
for f in np.linspace(10, 1000, 10):
    sinusoidal_current = input_factory.get_sinusoidal_current(200, 1000, unit_time=0.1 * b2.ms,
                                                              amplitude=2.5 * b2.namp, frequency=f*b2.Hz,
                                                              direct_current=10 * b2.namp, phase_offset=0.5)

    # run the LIF model.
    # Note: As we do not specify any model parameters, the simulation runs with the default values
    (state_monitor, spike_monitor) = LIF.simulate_LIF_neuron(
        input_current=step_current, simulation_time=100 * b2.ms)

    # plot I and vm
    plot_tools.plot_voltage_and_current_traces(
        state_monitor, sinusoidal_current, title="Sinusoidal input current")
    print("nr of spikes: {}".format(spike_monitor.count[0]))

# The Exponential Integrate-and-Fire model

input_current = input_factory.get_step_current(
    t_start=20, t_end=120, unit_time=b2.ms, amplitude=0.8 * b2.namp)

state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(
    I_stim=input_current, simulation_time=200*b2.ms)

plot_tools.plot_voltage_and_current_traces(
    state_monitor, input_current, title="step current",
    firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike)
print("nr of spikes: {}".format(spike_monitor.count[0]))

# 2.1. Exercise: rehobase threshold
MEMBRANE_TIME_SCALE_tau = 12.0 * b2.ms
MEMBRANE_RESISTANCE_R = 20.0 * b2.Mohm
V_REST = -65.0 * b2.mV
V_RESET = -60.0 * b2.mV
RHEOBASE_THRESHOLD_v_rh = -55.0 * b2.mV
SHARPNESS_delta_T = 2.0 * b2.mV
FIRING_THRESHOLD_v_spike = -30. * b2.mV
'''Modify the code example given above: Call simulate_exponential_IF_neuron() and set the function parameter v_spike to +10mV (which overrides the default value -30mV). What do you expect to happen? How many spikes will be generated?'''
state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(
    v_spike=10.0*b2.mV, I_stim=input_current, simulation_time=200*b2.ms)

plot_tools.plot_voltage_and_current_traces(
    state_monitor, input_current, title="step current",
    firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike)
print("nr of spikes: {}".format(spike_monitor.count[0]))
# nr of spikes: 7

'''Compute the minimal amplitude Irh
 of a constant input current such that the neuron will elicit a spike. If you are not sure what and how to compute Irh
, have a look at Figure 5.1 and the textbox “Rheobase threshold and interpretation of parameters” in the book.
'''
Iext = 10*b2.namp
f = -(V_RESET-V_REST)+SHARPNESS_delta_T * \
    np.exp((V_RESET-V_REST)/SHARPNESS_delta_T)
# I = du2dt - f/R
I = (f + MEMBRANE_RESISTANCE_R*Iext)/MEMBRANE_TIME_SCALE_tau
'''
Validate your result: Modify the code given above and inject a current of amplitude Irh
 and 300ms duration into the expIF neuron.
input_current = input_factory.get_step_current(
    t_start=20, t_end=120, unit_time=b2.ms, amplitude=0.8 * b2.namp)
'''
state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(
    v_spike=10.0*b2.mV, I_stim=input_current, simulation_time=300*b2.ms)

plot_tools.plot_voltage_and_current_traces(
    state_monitor, input_current, title="step current",
    firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike)
print("nr of spikes: {}".format(spike_monitor.count[0]))

# 2.12. Exercise: strength-duration curve
'''
Have a look at the following code: for the values i = 0, 2 and 6 we did not provide the minimal amplitude, but the entries in min_amp[i] are set to 0. Complete the min_amp list.
'''
i0 = 6  # change i and find the value that goes into min_amp
durations = [1,   2,    5,  10,   20,   50, 100]
min_amp = [0., 4.42, 10., 1.10, .70, .48, 5.0]
spike_number = []

I_amp = min_amp[i0]*b2.namp

for i in range(7):
    t = durations[i]
    title_txt = "I_amp={}, t={}".format(I_amp, t*b2.ms)

    input_current = input_factory.get_step_current(
        t_start=10, t_end=10+t-1, unit_time=b2.ms, amplitude=I_amp)

    state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(
        I_stim=input_current, simulation_time=(t+20)*b2.ms)

    plot_tools.plot_voltage_and_current_traces(state_monitor, input_current,
                                               title=title_txt, firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike,
                                               legend_location=2)
    spike_number.append(spike_monitor.count[0])
    print("nr of spikes: {}".format(spike_monitor.count[0]))
    # 0:0 4.42:0 1.1:0 10:5 5,2:1, :

plt.plot(durations, min_amp)
plt.title("Strength-Duration curve")
plt.xlabel("t [ms]")
plt.ylabel("min amplitude [nAmp]")

# 3. AdEx: the Adaptive Exponential Integrate-and-Fire model

current = input_factory.get_step_current(10, 250, 1. * b2.ms, 65.0 * b2.pA)
state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(
    I_stim=current, simulation_time=400 * b2.ms)
plot_tools.plot_voltage_and_current_traces(state_monitor, current)
print("nr of spikes: {}".format(spike_monitor.count[0]))

plt.figure()
plt.plot(state_monitor.v[0])
plt.title('u')
plt.figure()
plt.plot(state_monitor.w[0])
plt.title('w')


# AdEx.plot_adex_state(state_monitor)
# Use the terminology of Fig. 6.1 in Chapter 6.1.
# Call the function AdEx.simulate_AdEx_neuron() with different parameters and try to create adapting, bursting and irregular firing patterns.

# 7. Hopfield Network model of associative memory

pattern_size = 5

# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(nr_neurons=pattern_size**2)
# instantiate a pattern factory
factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
# create a checkerboard pattern and add it to the pattern list
checkerboard = factory.create_checkerboard()
pattern_list = [checkerboard]

# add random patterns to the list
pattern_list.extend(factory.create_random_pattern_list(
    nr_patterns=3, on_probability=0.5))
plot_tools.plot_pattern_list(pattern_list)
# how similar are the random patterns and the checkerboard? Check the overlaps
overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
plot_tools.plot_overlap_matrix(overlap_matrix)

# let the hopfield network "learn" the patterns. Note: they are not stored
# explicitly but only network weights are updated !
hopfield_net.store_patterns(pattern_list)

# create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.flip_n(checkerboard, nr_of_flips=4)
hopfield_net.set_state_from_pattern(noisy_init_state)

# from this initial state, let the network dynamics evolve.
states = hopfield_net.run_with_monitoring(nr_steps=4)

# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = factory.reshape_patterns(states)
# plot the states of the network
plot_tools.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")


MEMBRANE_TIME_SCALE_tau_m = 5 * b2.ms
MEMBRANE_RESISTANCE_R = 500*b2.Mohm
V_REST = -70.0 * b2.mV
V_RESET = -51.0 * b2.mV
RHEOBASE_THRESHOLD_v_rh = -50.0 * b2.mV
SHARPNESS_delta_T = 2.0 * b2.mV
ADAPTATION_VOLTAGE_COUPLING_a = 0.5 * b2.nS
ADAPTATION_TIME_CONSTANT_tau_w = 100.0 * b2.ms
SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b = 7.0 * b2.pA


# the letters we want to store in the hopfield network
letter_list = ['A', 'B', 'C', 'S', 'X', 'Y', 'Z']

# set a seed to reproduce the same noise in the next run
# numpy.random.seed(123)

abc_dictionary = pattern_tools.load_alphabet()
print("the alphabet is stored in an object of type: {}".format(type(abc_dictionary)))
# access the first element and get it's size (they are all of same size)
pattern_shape = abc_dictionary['A'].shape
print("letters are patterns of size: {}. Create a network of corresponding size".format(pattern_shape))
# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(
    nr_neurons=pattern_shape[0]*pattern_shape[1])

# create a list using Pythons List Comprehension syntax:
pattern_list = [abc_dictionary[key] for key in letter_list]
plot_tools.plot_pattern_list(pattern_list)

# store the patterns
hopfield_net.store_patterns(pattern_list)

# # create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.get_noisy_copy(
    abc_dictionary['A'], noise_level=0.2)
hopfield_net.set_state_from_pattern(noisy_init_state)

# from this initial state, let the network dynamics evolve.
states = hopfield_net.run_with_monitoring(nr_steps=4)

# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = pattern_tools.reshape_patterns(
    states, pattern_list[0].shape)

# plot the states of the network
plot_tools.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")

'''
The network is initialized with a (very) noisy pattern S(t=0)
. Then, the dynamics recover pattern P0 in 5 iterations.
Overlap matrix
array([[ 1.  , -0.36,  0.2 , -0.36],
       [-0.36,  1.  ,  0.12,  0.2 ],
       [ 0.2 ,  0.12,  1.  , -0.36],
       [-0.36,  0.2 , -0.36,  1.  ]])
'''


pattern_size = 5

# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(nr_neurons=pattern_size**2)
# instantiate a pattern factory
factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
# create a checkerboard pattern and add it to the pattern list
checkerboard = factory.create_checkerboard()
pattern_list = [checkerboard]

# add random patterns to the list
pattern_list.extend(factory.create_random_pattern_list(
    nr_patterns=3, on_probability=0.5))
plot_tools.plot_pattern_list(pattern_list)
# how similar are the random patterns and the checkerboard? Check the overlaps
overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
plot_tools.plot_overlap_matrix(overlap_matrix, color_map='bwr')

# let the hopfield network "learn" the patterns. Note: they are not stored
# explicitly but only network weights are updated !
hopfield_net.store_patterns(pattern_list)

# create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.flip_n(checkerboard, nr_of_flips=4)
hopfield_net.set_state_from_pattern(noisy_init_state)

# from this initial state, let the network dynamics evolve.
states = hopfield_net.run_with_monitoring(nr_steps=4)

# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = factory.reshape_patterns(states)
# plot the states of the network
plot_tools.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")

# using hfplot

'''
using unchanged overlap-matrix
'''
reference_pattern = 0
initially_flipped_pixels = 3
nr_iterations = 6
hfplot.plot_pattern_list(pattern_list)
# let the hopfield network "learn" the patterns. Note: they are not stored
# explicitly but only network weights are updated !
hopfield_net.store_patterns(pattern_list)

# how similar are the random patterns? Check the overlaps
overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
hfplot.plot_overlap_matrix(overlap_matrix, color_map='bwr')
# create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.flip_n(
    pattern_list[reference_pattern], initially_flipped_pixels)
hopfield_net.set_state_from_pattern(noisy_init_state)

# uncomment the following line to enable a PROBABILISTIC network dynamic
# hopfield_net.set_dynamics_probabilistic_sync(2.5)
# uncomment the following line to enable an ASYNCHRONOUS network dynamic
# hopfield_net.set_dynamics_sign_async()

# run the network dynamics and record the network state at every time step
states = hopfield_net.run_with_monitoring(nr_iterations)
# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = factory.reshape_patterns(states)
# plot the states of the network
hfplot.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, reference_pattern)
plt.show()


'''
using new overlap-matrix
array([[ 1.   ,  0.375,  0.25 , -0.125],
       [ 0.375,  1.   , -0.125,  0.   ],
       [ 0.25 , -0.125,  1.   ,  0.625],
       [-0.125,  0.   ,  0.625,  1.   ]])
'''
pattern_size = 4
nr_random_patterns = 3
reference_pattern = 0
initially_flipped_pixels = 3
nr_iterations = 5
random_seed = None
# instantiate a hofpfield network
hopfield_net = network.HopfieldNetwork(pattern_size**2)

# for the demo, use a seed to get a reproducible pattern
np.random.seed(random_seed)

# instantiate a pattern factory
factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
# create a checkerboard pattern and add it to the pattern list
checkerboard = factory.create_checkerboard()
pattern_list = [checkerboard]
# add random patterns to the list
pattern_list.extend(factory.create_random_pattern_list(
    nr_random_patterns, on_probability=0.5))
hfplot.plot_pattern_list(pattern_list)
# let the hopfield network "learn" the patterns. Note: they are not stored
# explicitly but only network weights are updated !
hopfield_net.store_patterns(pattern_list)

# how similar are the random patterns? Check the overlaps
overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
hfplot.plot_overlap_matrix(overlap_matrix, color_map='bwr')

# create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.flip_n(
    pattern_list[reference_pattern], initially_flipped_pixels)
hopfield_net.set_state_from_pattern(noisy_init_state)

# uncomment the following line to enable a PROBABILISTIC network dynamic
# hopfield_net.set_dynamics_probabilistic_sync(2.5)
# uncomment the following line to enable an ASYNCHRONOUS network dynamic
# hopfield_net.set_dynamics_sign_async()

# run the network dynamics and record the network state at every time step
states = hopfield_net.run_with_monitoring(nr_iterations)
# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = factory.reshape_patterns(states)
# plot the states of the network
hfplot.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, reference_pattern)
plt.show()


# 7.2. Introduction: Hopfield-networks
# Exercise: N=4x4 Hopfield-network
'''
7.3.1. Question: Storing a single pattern
Modify the Python code given above to implement this exercise:

Create a network with N=16
 neurons.
Create a single 4 by 4 checkerboard pattern.
Store the checkerboard in the network.
Set the initial state of the network to a noisy version of the checkerboard (nr_flipped_pixels = 5).
Let the network dynamics evolve for 4 iterations.
Plot the sequence of network states along with the overlap of network state with the checkerboard.
Now test whether the network can still retrieve the pattern if we increase the number of flipped pixels. What happens at nr_flipped_pixels = 8, what if nr_flipped_pixels > 8 ?
'''
'''
using unchanged overlap-matrix
'''
reference_pattern = 0
initially_flipped_pixels = 5
nr_iterations = 4
pattern_size = 4
random_seed = None

# instantiate a hofpfield network
hopfield_net = network.HopfieldNetwork(pattern_size**2)

# for the demo, use a seed to get a reproducible pattern
np.random.seed(random_seed)

# instantiate a pattern factory
factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
# create a checkerboard pattern and add it to the pattern list
checkerboard = factory.create_checkerboard()
pattern_list = [checkerboard]
# add random patterns to the list
pattern_list.extend(factory.create_random_pattern_list(
    nr_random_patterns, on_probability=0.5))
hfplot.plot_pattern_list(pattern_list)
# let the hopfield network "learn" the patterns. Note: they are not stored
# explicitly but only network weights are updated !
hopfield_net.store_patterns(pattern_list)

# how similar are the random patterns? Check the overlaps
overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
hfplot.plot_overlap_matrix(overlap_matrix, color_map='bwr')

hfplot.plot_pattern_list(pattern_list)
# let the hopfield network "learn" the patterns. Note: they are not stored
# explicitly but only network weights are updated !
hopfield_net.store_patterns(pattern_list)

# create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.flip_n(
    pattern_list[reference_pattern], initially_flipped_pixels)
hopfield_net.set_state_from_pattern(noisy_init_state)

# uncomment the following line to enable a PROBABILISTIC network dynamic
# hopfield_net.set_dynamics_probabilistic_sync(2.5)
# uncomment the following line to enable an ASYNCHRONOUS network dynamic
# hopfield_net.set_dynamics_sign_async()

# run the network dynamics and record the network state at every time step
states = hopfield_net.run_with_monitoring(nr_iterations)
# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = factory.reshape_patterns(states)
# plot the states of the network
hfplot.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, reference_pattern)
plt.show()

# 7.3.2. Question: the weights matrix
# The patterns a Hopfield network learns are not stored explicitly. Instead, the network learns by adjusting the weights to the pattern set it is presented during learning. Let’s visualize this.
pattern_size = 4
# Create a new 4x4 network. Do not yet store any pattern.
# instantiate a hofpfield network
hopfield_net = network.HopfieldNetwork(pattern_size**2)
# What is the size of the network matrix?
# Visualize the weight matrix using the function plot_tools.plot_nework_weights(). It takes the network as a parameter.
plot_tools.plot_network_weights(hopfield_net)
# Create a checkerboard, store it in the network.
factory1 = pattern_tools.PatternFactory(pattern_size, pattern_size)
# create a checkerboard pattern and add it to the pattern list
checkerboard1 = factory1.create_checkerboard()
pattern_list1 = [checkerboard1]
# add random patterns to the list
#pattern_list.extend(factory.create_random_pattern_list(
#    nr_random_patterns, on_probability=0.5))
hfplot.plot_pattern_list(pattern_list1)
# let the hopfield network "learn" the patterns. Note: they are not stored
# explicitly but only network weights are updated !
hopfield_net.store_patterns(pattern_list1)
# Plot the weights matrix. What weight values do occur?
plot_tools.plot_network_weights(hopfield_net)
# Create a new 4x4 network.
pattern_size = 4
# Create a new 4x4 network. Do not yet store any pattern.
# instantiate a hofpfield network
hopfield_net = network.HopfieldNetwork(pattern_size**2)
factory2 = pattern_tools.PatternFactory(pattern_size, pattern_size)
# Create an L-shaped pattern (look at pattern_tools.PatternFactory), store it in the network.
Lshaped = factory2.create_L_pattern()
pattern_list2 = [Lshaped]
hfplot.plot_pattern_list(pattern_list1)
# let the hopfield network "learn" the patterns. Note: they are not stored
# explicitly but only network weights are updated !
hopfield_net.store_patterns(pattern_list2)
# Plot the weights matrix. What weight values do occur?
plot_tools.plot_network_weights(hopfield_net)
# Create a new 4x4 network.
pattern_size = 4
# Create a new 4x4 network. Do not yet store any pattern.
# instantiate a hofpfield network
hopfield_net = network.HopfieldNetwork(pattern_size**2)
factory3 = pattern_tools.PatternFactory(pattern_size, pattern_size)
# Create a checkerboard and an L-shaped pattern. Store both patterns in the network.
checkerboard3 = factory3.create_checkerboard()
Lshaped3 = factory3.create_L_pattern()
pattern_list3=[checkerboard3,Lshaped3]
hopfield_net.store_patterns(pattern_list3)
# Plot the weights matrix. What weight values do occur? How does this matrix compare to the two previous matrices?
plot_tools.plot_network_weights(hopfield_net)
#7.3.3. Question (optional): Weights Distribution
#It’s interesting to look at the weights distribution in the three previous cases. You can easily plot a histogram by adding the following two lines to your script. It assumes you have stored your network in the variable hopfield_net.
plt.figure()
plt.hist(hopfield_net.weights.flatten())
'''
(array([ 64.,   0.,   0.,   0.,   0., 142.,   0.,   0.,   0.,  50.]),
 array([-0.125, -0.1  , -0.075, -0.05 , -0.025,  0.   ,  0.025,  0.05 ,
         0.075,  0.1  ,  0.125]),
 <BarContainer object of 10 artists>)
'''
# 7.4. Exercise: Capacity of an N=100 Hopfield-network
'''
For a large number of steps, the positive or negative walking distance can be approximated by a
Gaussian distribution with zero mean and standard deviation σ =(M −1)/N ≈
M/N,for M ≫ 1.
'''
#7.4.1. Question:
#A Hopfield network implements so called associative or content-adressable memory. Explain what this means.
'''
In the context of associative memories (to be discussed in the next
chapter) energy functions have been used for binary neuron models by Hopfield (1982)
and for rate models by Hopfield (1984).
For example, if we accept
an error probability of Perror = 0.001, we find a storage capacity of Cstore = 0.105.
Hence, a network of 10 000 neurons is capable of storing about 1000 patterns with
Perror = 0.001. Thus in each of the patterns, we expect that about 10 neurons exhibit
erroneous activity. We emphasize that the above calculation focuses on the first iteration
step only. If we start in the pattern, then about 10 neurons will flip their state in the first
iteration. But these flips could in principle cause further neurons to flip in the second
iteration and eventually initiate an avalanche of many other changes.
17.2.5 17.2.6
'''

#7.4.2. Question:
'''
we accept
an error probability of Perror = 0.001, we find a storage capacity of Cstore = 0.105.
Hence, a network of 10 000 neurons is capable of storing about 1000 patterns with
Perror = 0.001. Thus in each of the patterns, we expect that about 10 neurons exhibit
erroneous activity. We emphasize that the above calculation focuses on the first iteration
step only. If we start in the pattern, then about 10 neurons will flip their state in the first
iteration. But these flips could in principle cause further neurons to flip in the second
iteration and eventually initiate an avalanche of many other changes.
'''
'''Using the value Cstore
 given in the book, how many patterns can you store in a N=10x10 network? Use this number K
 in the next question:'''
 
'''The most important insight is that the probability of an erroneous state flip increases with
the ratio M/N. Formally, we can define the storage capacity Cstore of a network as the
maximal number Mmax of patterns that a network of N neurons can retrieve'''

'''Cstore = Mmax/N= Mmax/N2'''

'''
Since each pattern con-
sists of N neurons (i.e., N binary numbers), the total number of bits that need to be stored at
maximum capacity is Mmax N. In the Hopfield model, patterns are stored by an appropriate
choice of the synaptic connections. The number of available synapses in a fully connected
network is N2. Therefore, the storage capacity measures the number of bits stored per
synapse.
'''

'''!!!!!if we accept an error probability of Perror = 0.001, 
we find a storage capacity of Cstore = 0.105.
Hence, a network of 10000 neurons is capable of storing about 1000 patterns with
Perror = 0.001.'''

# 7.4.3. Question:
# Create an N=10x10 network and store a checkerboard pattern together with (K−1)
# random patterns. Then initialize the network with the unchanged checkerboard pattern. Let the network evolve for five iterations.
reference_pattern = 0
initially_flipped_pixels = 10
nr_iterations = 5
pattern_size = 10
random_seed = None
nr_random_patterns = 999 #K-1

# instantiate a hofpfield network
hopfield_net = network.HopfieldNetwork(pattern_size**2)

# for the demo, use a seed to get a reproducible pattern
np.random.seed(random_seed)

# instantiate a pattern factory
factory = pattern_tools.PatternFactory(pattern_size, pattern_size)
# create a checkerboard pattern and add it to the pattern list
checkerboard = factory.create_checkerboard()
pattern_list = [checkerboard]
# add random patterns to the list
pattern_list.extend(factory.create_random_pattern_list(
    nr_random_patterns, on_probability=0.1))
hfplot.plot_pattern_list(pattern_list)
# let the hopfield network "learn" the patterns. Note: they are not stored
# explicitly but only network weights are updated !
hopfield_net.store_patterns(pattern_list)

# how similar are the random patterns? Check the overlaps
overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
hfplot.plot_overlap_matrix(overlap_matrix, color_map='bwr')

# create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.flip_n(
    pattern_list[reference_pattern], initially_flipped_pixels)
hopfield_net.set_state_from_pattern(noisy_init_state)

# uncomment the following line to enable a PROBABILISTIC network dynamic
# hopfield_net.set_dynamics_probabilistic_sync(2.5)
# uncomment the following line to enable an ASYNCHRONOUS network dynamic
# hopfield_net.set_dynamics_sign_async()

# run the network dynamics and record the network state at every time step
states = hopfield_net.run_with_monitoring(nr_iterations)
# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = factory.reshape_patterns(states)
# plot the states of the network
hfplot.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, reference_pattern)
plt.show()

# Rerun your script a few times. What do you observe?
#connectivity exists at same columns but with different value

# 7.5. Exercise: Non-random patterns
# In the previous exercises we used random patterns. Now we us a list of structured patterns: the letters A to Z. Each letter is represented in a 10 by 10 grid.

# Eight letters (including ‘A’) are stored in a Hopfield network. The letter ‘A’ is not recovered.

# 7.5.1. Question:
# Run the following code. Read the inline comments and look up the doc of functions you do not know.


# the letters we want to store in the hopfield network
letter_list = ['A', 'B', 'C', 'S', 'X', 'Y', 'Z']

# set a seed to reproduce the same noise in the next run
# numpy.random.seed(123)

abc_dictionary = pattern_tools.load_alphabet()
print("the alphabet is stored in an object of type: {}".format(type(abc_dictionary)))
# access the first element and get it's size (they are all of same size)
pattern_shape = abc_dictionary['A'].shape
print("letters are patterns of size: {}. Create a network of corresponding size".format(pattern_shape))
# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(
    nr_neurons=pattern_shape[0]*pattern_shape[1])

# create a list using Pythons List Comprehension syntax:
pattern_list = [abc_dictionary[key] for key in letter_list]
plot_tools.plot_pattern_list(pattern_list)

# store the patterns
hopfield_net.store_patterns(pattern_list)

# # create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.get_noisy_copy(
    abc_dictionary['A'], noise_level=0.2)
hopfield_net.set_state_from_pattern(noisy_init_state)

# from this initial state, let the network dynamics evolve.
states = hopfield_net.run_with_monitoring(nr_steps=4)

# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = pattern_tools.reshape_patterns(
    states, pattern_list[0].shape)

# plot the states of the network
plot_tools.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")

#7.5.2. Question:
#Add the letter ‘R’ to the letter list and store it in the network. Is the pattern ‘A’ still a fixed point? Does the overlap between the network state and the reference pattern ‘A’ always decrease?
# the letters we want to store in the hopfield network
letter_list = ['A', 'B', 'C','R', 'S', 'X', 'Y', 'Z']

# set a seed to reproduce the same noise in the next run
# numpy.random.seed(123)

abc_dictionary = pattern_tools.load_alphabet()
print("the alphabet is stored in an object of type: {}".format(type(abc_dictionary)))
# access the first element and get it's size (they are all of same size)
pattern_shape = abc_dictionary['A'].shape
print("letters are patterns of size: {}. Create a network of corresponding size".format(pattern_shape))
# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(
    nr_neurons=pattern_shape[0]*pattern_shape[1])

# create a list using Pythons List Comprehension syntax:
pattern_list = [abc_dictionary[key] for key in letter_list]
plot_tools.plot_pattern_list(pattern_list)

# store the patterns
hopfield_net.store_patterns(pattern_list)

# # create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.get_noisy_copy(
    abc_dictionary['A'], noise_level=0.2)
hopfield_net.set_state_from_pattern(noisy_init_state)

# from this initial state, let the network dynamics evolve.
states = hopfield_net.run_with_monitoring(nr_steps=4)

# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = pattern_tools.reshape_patterns(
    states, pattern_list[0].shape)

# plot the states of the network
plot_tools.plot_state_sequence_and_overlap(
    states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")

#7.5.3. Question:
#Make a guess of how many letters the network can store. Then create a(small) set of letters. Check if all letters of your list are fixed points under the network dynamics. Explain the discrepancy between the network capacity C
#(computed above) and your observation.
M = 10*10#(shape of pattern A)
N = 2**6 #7 bits if add R successfully
C= N/M #0.64
#7.6. Exercise: Bonus
#The implementation of the Hopfield Network in hopfield_network.network offers a possibility to provide a custom update function HopfieldNetwork.set_dynamics_to_user_function(). Have a look at the source code of HopfieldNetwork.set_dynamics_sign_sync() to learn how the update dynamics are implemented. Then try to implement your own function. For example, you could implement an asynchronous update with stochastic neurons.
"""
This file implements a Hopfield network. It provides functions to
set and retrieve the network state, store patterns.

Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch17.S2.html

"""

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

import numpy as np
import neurodynex3.hopfield_network


class HopfieldNetwork:
    """Implements a Hopfield network.

    Attributes:
        nrOfNeurons (int): Number of neurons
        weights (numpy.ndarray): nrOfNeurons x nrOfNeurons matrix of weights
        state (numpy.ndarray): current network state. matrix of shape (nrOfNeurons, nrOfNeurons)
    """

    def __init__(self, nr_neurons):
        """
        Constructor

        Args:
            nr_neurons (int): Number of neurons. Use a square number to get the
            visualizations properly
        """
        # math.sqrt(nr_neurons)
        self.nrOfNeurons = nr_neurons
        # initialize with random state
        self.state = 2 * np.random.randint(0, 2, self.nrOfNeurons) - 1
        # initialize random weights
        self.weights = 0
        self.reset_weights()
        self._update_method = _get_sign_update_function()

    def reset_weights(self):
        """
        Resets the weights to random values
        """
        self.weights = 1.0 / self.nrOfNeurons * \
            (2 * np.random.rand(self.nrOfNeurons, self.nrOfNeurons) - 1)


    def set_dynamics_sign_sync(self):
        """
        sets the update dynamics to the synchronous, deterministic g(h) = sign(h) function
        """
        self._update_method = _get_sign_update_function()


    def set_dynamics_sign_async(self):
        """
        Sets the update dynamics to the g(h) =  sign(h) functions. Neurons are updated asynchronously:
        In random order, all neurons are updated sequentially
        """
        self._update_method = _get_async_sign_update_function()


    def set_dynamics_to_user_function(self, update_function):
        """
        Sets the network dynamics to the given update function

        Args:
            update_function: upd(state_t0, weights) -> state_t1.
                Any function mapping a state s0 to the next state
                s1 using a function of s0 and weights.
        """
        self._update_method = update_function


    def store_patterns(self, pattern_list):
        """
        Learns the patterns by setting the network weights. The patterns
        themselves are not stored, only the weights are updated!
        self connections are set to 0.

        Args:
            pattern_list: a nonempty list of patterns.
        """
        all_same_size_as_net = all(len(p.flatten()) == self.nrOfNeurons for p in pattern_list)
        if not all_same_size_as_net:
            errMsg = "Not all patterns in pattern_list have exactly the same number of states " \
                     "as this network has neurons n = {0}.".format(self.nrOfNeurons)
            raise ValueError(errMsg)
        self.weights = np.zeros((self.nrOfNeurons, self.nrOfNeurons))
        # textbook formula to compute the weights:
        for p in pattern_list:
            p_flat = p.flatten()
            for i in range(self.nrOfNeurons):
                for k in range(self.nrOfNeurons):
                    self.weights[i, k] += p_flat[i] * p_flat[k]
        self.weights /= self.nrOfNeurons
        # no self connections:
        np.fill_diagonal(self.weights, 0)


    def set_state_from_pattern(self, pattern):
        """
        Sets the neuron states to the pattern pixel. The pattern is flattened.

        Args:
            pattern: pattern
        """
        self.state = pattern.copy().flatten()


    def iterate(self):
        """Executes one timestep of the dynamics"""
        self.state = self._update_method(self.state, self.weights)


    def run(self, nr_steps=5):
        """Runs the dynamics.

        Args:
            nr_steps (float, optional): Timesteps to simulate
        """
        for i in range(nr_steps):
            # run a step
            self.iterate()


    def run_with_monitoring(self, nr_steps=5):
        """
        Iterates at most nr_steps steps. records the network state after every
        iteration

        Args:
            nr_steps:

        Returns:
            a list of 2d network states
        """
        states = list()
        states.append(self.state.copy())
        for i in range(nr_steps):
            # run a step
            self.iterate()
            states.append(self.state.copy())
        return states

    def _get_sign_update_function():
        """
        for internal use
    
        Returns:
            A function implementing a synchronous state update using sign(h)
        """
        def upd(state_s0, weights):
            h = np.sum(weights * state_s0, axis=1)
            s1 = np.sign(h)
            # by definition, neurons have state +/-1. If the
            # sign function returns 0, we set it to +1
            idx0 = s1 == 0
            s1[idx0] = 1
            return s1
        return upd
    
    
    def _get_async_sign_update_function():
        def upd(state_s0, weights):
            random_neuron_idx_list = np.random.permutation(len(state_s0))
            state_s1 = state_s0.copy()
            for i in range(len(random_neuron_idx_list)):
                rand_neuron_i = random_neuron_idx_list[i]
                h_i = np.dot(weights[:, rand_neuron_i], state_s1)
                s_i = np.sign(h_i)
                if s_i == 0:
                    s_i = 1
                state_s1[rand_neuron_i] = s_i
            return state_s1
        return upd


#if __name__ == "__main__":
#    neurodynex3.hopfield_network.demo.run_demo()