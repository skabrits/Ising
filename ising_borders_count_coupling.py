import numpy as np
import matplotlib.pyplot as plt
import tqdm as tq
import sys


def find_coupling(spin_lattice):
    h, w = spin_lattice.shape
    i, j = (1, 0)
    try:
        i, j = step_direction(spin_lattice, i, j, 1)
    except Exception as e:
        plt.imshow(spin_lattice)
        plt.savefig('debug.png')
        print(e)
        raise Exception("recursion limit exceeded")
    if i >= h-1:
        return 1
    else:
        return -1


def step_direction(spin_lattice, i0, j0, direction):
    h, w = spin_lattice.shape
    step = direction*1j
    i, j = (i0, j0)
    s0 = spin_lattice[i, j]
    if i < h-1 and j < w-1:
        if spin_lattice[i+int(np.real(step)), j+int(np.imag(step))] == s0:
            i, j = i+int(np.real(step)), int(j+np.imag(step))
            return step_direction(spin_lattice, i, j, step)
        elif spin_lattice[i+int(np.real(direction)), j+int(np.imag(direction))] == s0:
            i, j = i+int(np.real(direction)), j+int(np.imag(direction))
            return step_direction(spin_lattice, i, j, direction)
        else:
            step = direction*(-1j)
            if spin_lattice[i+int(np.real(step)), j+int(np.imag(step))] == s0:
                i, j = i+int(np.real(step)), j+int(np.imag(step))
                return step_direction(spin_lattice, i, j, step)
            else:
                i, j = i+int(np.real(-direction)), j+int(np.imag(-direction))
                return step_direction(-spin_lattice, i, j, -direction)
    else:
        return i, j


def switch(i, j, spin_lattice, T, W=(1,1,1,1,1), J=1):
    h, w = spin_lattice.shape
    if i % h != 0 and i % h != h-1 and j % w != 0 and j % w != w-1:
        h, w = spin_lattice.shape
        e = (spin_lattice[(i-1) % h, j % w] + spin_lattice[(i+1) % h, j % w] + spin_lattice[i % h, (j-1) % w]+spin_lattice[i % h, (j+1) % w])
        E = spin_lattice[i % h, j % w]*e*(-J)
        nE = -spin_lattice[i % h, j % w]*e*(-J)
        p = W[nE-E]
        if np.random.choice([0, 1], 1, p=[1-p, p]) == 1:
            return -spin_lattice[i % h, j % w]

    return spin_lattice[i % h, j % w]


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


def most_probable(T, size, iter=1000, R=5, graph=True, start_lattice=None, J=1):
    spin_lattice = np.pad(np.random.randint(2, size=size)*2 - 1, ((1, 1), (1, 1)), mode='constant', constant_values=((1, 1), (-1, -1))) if start_lattice is None else start_lattice
    coup = 0
    ac = 0
    W = {i: (np.exp(-i/T) if T != 0 else 0) if i > 0 else 1 for i in range(-8, 9, 4)}

    if graph:
        im = plt.imshow(spin_lattice)
        plt.show(block=False)
        plt.pause(0.5)

    for k in range(iter):
        i0 = np.random.randint(1, size[0]+1)
        j0 = np.random.randint(1, size[1]+1)

        s = switch(i0, j0, spin_lattice, T, W=W, J=J)

        if spin_lattice[i0, j0] != s:
            spin_lattice[i0, j0] = s
            for r in range(1, R):
                spin_lattice, end = switch_radius(i0, j0, r, spin_lattice, T, W=W, J=J)
                if end:
                    break

        if k >= 0.5*iter:
            coup += find_coupling(spin_lattice)
            ac += 1
        if graph:
            if k % round(1000/R) == 0:
                plt.pause(0.000001)
                im.set_data(spin_lattice)
                plt.draw()
    if graph:
        plt.pause(1)
        plt.close()

    return coup/ac if ac > 5 else find_coupling(spin_lattice)


if __name__ == '__main__':
    base_size = 10                  # base lattice size
    J = 1                           # energy constant; should include k_b; ferromagnetic if positive, antiferromagneic else
    R = 3                           # wave propagation radius of simulation
    visualize_simulation = False    # draw lattice during simulation
    T_0 = 2.69                      # critical temperature
    G_0 = 0.5                       # zero aspect ratio
    G_1 = 4                        # end aspect ratio
    N = 36                         # number of temperatures
    sim_iter = 30000                # number of iterations for simulation

    sys.setrecursionlimit(max(sys.getrecursionlimit(), base_size**2*G_1))
    ar = np.linspace(G_0, G_1, N)
    coupling = list()
    for ratio in tq.tqdm(ar):
        size = np.array([base_size, base_size*ratio], dtype=int)
        mp = most_probable(T_0, size, iter=sim_iter, R=R, graph=visualize_simulation, J=J)
        coupling.append(mp)
    f, ax1 = plt.subplots(1, sharex=True)
    ax1.plot(ar, coupling)
    ax1.set_title('coupling, T = ' + str(T_0))
    ax1.set_xscale('log')
    plt.show()
