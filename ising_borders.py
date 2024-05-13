import numpy as np
import matplotlib.pyplot as plt
import tqdm as tq
import matplotlib.animation as animation


def count_energy(state, J=1):
    h, w = state.shape
    energy = 0
    for i in range(0, h):
        for j in range(0, w):
            energy += -J * state[i, j] * (state[(i-1) % h, j] + state[(i+1) % h, j] + state[i, (j-1) % w]+state[i, (j+1) % w])
    return energy/2


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


def most_probable(T, size, iter=1000, R=5, graph=True, start_lattice=None, J=1):
    spin_lattice = np.pad(np.random.randint(2, size=size)*2 - 1, ((1, 1), (1, 1)), mode='constant', constant_values=((1, 1), (-1, -1))) if start_lattice is None else start_lattice
    avm = np.zeros(shape=(size+2))
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
            avm += spin_lattice
            ac += 1
        if graph:
            if k % round(1000/R) == 0:
                plt.pause(0.000001)
                im.set_data(spin_lattice)
                plt.draw()
    if graph:
        plt.pause(1)
        plt.close()

    return avm/ac if ac > 5 else spin_lattice, spin_lattice


def animate(i):
    im.set_data(sm[i])
    ax1.set_title('<|s|>, T = ' + str(temperatures[i]))
    return im,


if __name__ == '__main__':
    size = np.array([50, 50])        # lattice size
    J = 1                            # energy constant; should include k_b; ferromagnetic if positive, antiferromagneic else
    R = 3                            # wave propagation radius of simulation
    visualize_simulation = False     # draw lattice during simulation
    T_0 = 0.00001                    # zero temperature
    T_1 = 10                         # end temperature
    N = 50                           # number of temperatures
    sim_iter = 100000                 # number of iterations for simulation

    temperatures = np.linspace(T_0, T_1, N)
    start_lattice = np.pad(np.ones(shape=size), ((1, 1), (1, 1)), mode='constant', constant_values=((1, 1), (-1, -1)))
    plt.imshow(start_lattice)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    sm = list()
    for t in tq.tqdm(temperatures):
        mp = most_probable(t, size, iter=sim_iter, R=R, graph=visualize_simulation, start_lattice=start_lattice, J=J)
        start_lattice = mp[1]
        sm.append(mp[0])
    f, ax1 = plt.subplots(1, sharex=True)
    im = ax1.imshow(np.pad(np.ones(shape=size), ((1, 1), (1, 1)), mode='constant', constant_values=((1, 1), (-1, -1))))
    ax1.set_title('<|s|>')

    ani = animation.FuncAnimation(f, animate, repeat=True, frames=len(sm) - 1, interval=1000)
    writer = animation.PillowWriter(fps=1, bitrate=1800)
    ani.save('50x50x100000.gif', writer=writer)
    plt.show()
