import numpy as np
import matplotlib.pyplot as plt
import tqdm as tq


def calculate_average_spin(state):
    return np.abs(np.sum(state))/np.prod(state.shape)


def calculate_average_spin_scalar(state, r):
    h, w = state.shape
    return np.sum([scalar_radius(i, j, state, r=r) for i in range(h) for j in range(w)])/(4*r*h*w)


def count_energy(state, J=1):
    h, w = state.shape
    energy = 0
    for i in range(0, h):
        for j in range(0, w):
            energy += -J * state[i, j] * (state[(i-1) % h, j] + state[(i+1) % h, j] + state[i, (j-1) % w]+state[i, (j+1) % w])
    return energy/2


def switch(i, j, spin_lattice, T, W=(1,1,1,1,1), J=1):
    h, w = spin_lattice.shape
    e = (spin_lattice[(i-1) % h, j % w] + spin_lattice[(i+1) % h, j % w] + spin_lattice[i % h, (j-1) % w]+spin_lattice[i % h, (j+1) % w])
    E = spin_lattice[i % h, j % w]*e*(-J)
    nE = -spin_lattice[i % h, j % w]*e*(-J)
    p = W[nE-E]
    if np.random.choice([0, 1], 1, p=[1-p, p]) == 1:
        return -spin_lattice[i % h, j % w]

    return spin_lattice[i % h, j % w]


def calculate_radius(i0, j0, i, j):
    return np.abs(i-i0)+np.abs(j-j0)


def calculate_second_index(i0, j0, ind, r, change_first=False):
    return [i0 + r - np.abs(ind-j0), i0 - r + np.abs(ind-j0)] if change_first else [j0 + r - np.abs(ind-i0), j0 - r + np.abs(ind-i0)]


def switch_radius(i0, j0, r, spin_lattice, T, W=(1, 1, 1, 1, 1), J=1):
    h, w = spin_lattice.shape
    i = i0 + r
    j = [j0]
    no_change = True
    while i > i0 - r - 1:
        for j_c in j:
            s = switch(i, j_c, spin_lattice, T, W=W, J=J)
            no_change = no_change and spin_lattice[i % h, j_c % w] == s
            spin_lattice[i % h, j_c % w] = s
        i -= 1
        j = calculate_second_index(i0, j0, i, r)
        if j[0] == j[1]:
            j.pop()

    return spin_lattice, no_change


def scalar_radius(i0, j0, spin_lattice, r=1):
    h, w = spin_lattice.shape
    i = i0 + r
    j = [j0]
    scalar_sum = 0
    while i > i0 - r - 1:
        for j_c in j:
            scalar_sum += spin_lattice[i0, j0] * spin_lattice[i % h, j_c % w]
        i -= 1
        j = calculate_second_index(i0, j0, i, r)
        if j[0] == j[1]:
            j.pop()

    return scalar_sum


def most_probable(T, size, iter=1000, R=5, correlation_radius=1, graph=True, start_lattice=None, J=1):
    spin_lattice = np.random.randint(2, size=size)*2 - 1 if start_lattice is None else start_lattice
    avs = 0
    avss = 0
    ave = 0
    ac = 0
    W = {i: (np.exp(-i/T) if T != 0 else 0) if i > 0 else 1 for i in range(-8, 9, 4)}

    if graph:
        im = plt.imshow(spin_lattice)
        plt.show(block=False)
        plt.pause(0.5)

    for k in range(iter):
        i0 = np.random.randint(0, size[0])
        j0 = np.random.randint(0, size[1])

        s = switch(i0, j0, spin_lattice, T, W=W, J=J)

        if spin_lattice[i0, j0] != s:
            spin_lattice[i0, j0] = s
            for r in range(1, R):
                spin_lattice, end = switch_radius(i0, j0, r, spin_lattice, T, W=W, J=J)
                if end:
                    break

        if k >= 0.99*iter:
            tmp = calculate_average_spin(spin_lattice)
            avs += tmp
            avss += calculate_average_spin_scalar(spin_lattice, correlation_radius) - tmp**2
            ave += count_energy(spin_lattice, J=J)
            ac += 1
        if graph:
            if k % round(1000/R) == 0:
                plt.pause(0.000001)
                im.set_data(spin_lattice)
                plt.draw()
    if graph:
        plt.pause(1)
        plt.close()

    return avs/ac if ac > 5 else calculate_average_spin(spin_lattice), calculate_average_spin(spin_lattice), count_energy(spin_lattice), spin_lattice, avss/ac if ac > 5 else calculate_average_spin_scalar(spin_lattice, correlation_radius), calculate_average_spin_scalar(spin_lattice, correlation_radius), ave/ac if ac > 5 else count_energy(spin_lattice, J=J)


if __name__ == '__main__':
    size = np.array([50, 50])        # lattice size
    CR = 1                           # radius of 2 spin correlation calculation
    J = 1                            # energy constant; should include k_b; ferromagnetic if positive, antiferromagneic else
    R_0 = 5                          # wave propagation radius of start lattice calculation
    R_1 = 3                          # wave propagation radius of simulation
    visualize_start_lattice = False  # draw lattice during start lattice calculation
    visualize_simulation = False     # draw lattice during simulation
    T_0 = 0.00001                    # zero temperature
    T_1 = 10                         # end temperature
    N = 50                           # number of temperatures
    start_iter = 500000              # number of iterations for start lattice calculation
    sim_iter = 30000                 # number of iterations for simulation

    temperatures = np.linspace(T_0, T_1, N)
    start_lattice = most_probable(T_0, size, iter=start_iter, correlation_radius=CR, R=R_0, graph=visualize_start_lattice, J=J)[3]
    plt.imshow(start_lattice)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    spins = list()
    sc = list()
    e = list()
    for t in tq.tqdm(temperatures):
        mp = most_probable(t, size, iter=sim_iter, correlation_radius=CR, R=R_1, graph=visualize_simulation, start_lattice=start_lattice, J=J)
        start_lattice = mp[3]
        spins.append(mp[0])
        sc.append(mp[4])
        e.append(mp[6])
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(temperatures, spins)
    ax1.set_title('<|s|>')
    ax2.plot(temperatures, sc)
    ax2.set_title('<s_i*s_j> - <s_i><s_j>')
    ax3.plot(temperatures, e)
    ax3.set_title('E')
    plt.show()
