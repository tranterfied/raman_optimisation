## Optimisation code for calling the raman memory simulation
import subprocess
import xml.dom.minidom as xmlmd
import h5py
import numpy as np
import re
import ff
import pygmo as pg


class RamanProb:
    def __init__(self, dims):
        self.dims = dims

    def fitness(self, X):
        param_dict = {'gtin': str(X[0]),
                      'gpulsewidth_p': str(X[1]),
                      'gpulsewidth_m': str(X[1]),
                      'omega_in_p': str(X[2]),
                      'omega_in_m': str(X[3])
                      }

        cost = cost_function(param_dict)
        return [cost]

    def get_bounds(self):
        return [0, 0, 0, 0], [3.0, 3.0, 10, 20]


# Function to set all of the values in the simulation according to the dictionary
def set_params(xml_args, arg_dict):
    for k in arg_dict.keys():
        for arg in xml_args:
            arg_name = arg.getAttribute("name")
            if arg_name == k:
                arg.setAttribute("default_value", arg_dict[k])


# sets the new cdata in the file
def set_cdata(new_data={}):
    f = open('./sim_files/raman.xmds', 'r')
    lines = ''
    for line in f:
        new_line = line
        m = re.search('if \(time < \d+\.\d+\)', line)
        if m is not None:
            new_line = "if (time < {0:3.2f})".format(new_data.get('write_time', 2.4)) + "{ return 1.0; }"
        lines += new_line

    f.close()

    f = open('./sim_files/raman.xmds', 'w+')
    f.write(lines)
    f.close()


# evaluate how well the memory performed
def evaluate_performance():
    # open the data_file
    f = h5py.File('./raman.h5', 'r')

    # get the electric field array (forward)
    e_array = np.zeros(f['1']['EpI'].shape, dtype=np.complex)
    e_array.real = f['1']['EpR']
    e_array.imag = f['1']['EpI']

    # get the electric field array (backward)
    em_array = np.zeros(f['1']['EmI'].shape, dtype=np.complex)
    em_array.real = f['1']['EmR']
    em_array.imag = f['1']['EmI']

    f.close()

    # get the input and output pulses
    in_pulse = np.trapz((np.abs(e_array)**2)[:, 0])
    out_pulse = np.trapz((np.abs(em_array)**2)[:, 0])

    print('Efficiency: %s' % (out_pulse/in_pulse))
    return out_pulse/in_pulse


def cost_function(arg_dict):
    # open the XML document for parsing
    doc = xmlmd.parse('./base_sim_backward_simplified.xmds')

    # get all the arguments
    args = doc.getElementsByTagName('argument')

    # set the new arguments
    set_params(args, arg_dict)

    # open the file and write it
    new_file = open('./sim_files/raman.xmds', 'w+')
    doc.writexml(new_file)
    new_file.close()

    # set the cdata that is not in XML format
    set_cdata(arg_dict)

    # run xmds to generate the new file
    # print('Generating simulation code...', end='')
    xmds_proc = subprocess.run(['xmds2', './sim_files/raman.xmds'], capture_output=True)
    # print(' Done.')

    # run the simulation
    # print('Running the simulation...', end='')
    try:
        sim_proc = subprocess.run(['./raman'], capture_output=True, timeout=10)
    except subprocess.TimeoutExpired:
        print('Timed out during simulation')
        return 1.0
    # print(' Done.')

    # return the loss so we minimise it
    cost = evaluate_performance()
    # print(arg_dict)
    return 1 - cost

if __name__ == '__main__':
    prob = pg.problem(RamanProb(4))
    pop = pg.population(prob=prob, size=10)

    iterations = 200
    fly = ff.fruit_fly(1e-5)
    # fly.x_0 = [2.4, 63]

    sols = []
    params = []
    for i in range(int(iterations / 10)):
        fly.evolve(pop)
        sols += [pop.champion_f]
        params += [pop.champion_x]

        if i % 2 == 0:
            print(i * 10, sols[-1], params[-1])

    print(np.min(sols), params[np.argmin(sols)])