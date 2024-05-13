import numpy as np
import matplotlib.pyplot as plt
import tqdm as tq

kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def calculate_average_spin(state):
    return np.abs(np.sum(state))/np.prod(np.array(state.shape)-2)


def count_energy(state):
    h, w = state.shape
    energy = 0
    for i in range(1, h-1):
        for j in range(1, w-1):
            energy += np.sum(state[i, j]*kernel*state[i-1:i+2, j-1:j+2])
    return energy/2


def switch(i, j, spin_lattice, T):
    E = np.sum(spin_lattice[i, j]*kernel*spin_lattice[i-1:i+2, j-1:j+2])
    nE = np.sum((-spin_lattice[i, j])*kernel*spin_lattice[i-1:i+2, j-1:j+2])
    if nE > E:
        return -spin_lattice[i, j]
    else:
        p = np.exp((nE-E)/T) if T != 0 else 0
        if np.random.choice([0, 1], 1, p=[1-p, p]) == 1:
            return -spin_lattice[i, j]

    return spin_lattice[i, j]


def calculate_radius(i0, j0, i, j):
    return np.abs(i-i0)+np.abs(j-j0)


def calculate_second_index(i0, j0, ind, r, change_first=False):
    return [i0 + r - np.abs(ind-j0), i0 - r + np.abs(ind-j0)] if change_first else [j0 + r - np.abs(ind-i0), j0 - r + np.abs(ind-i0)]


def switch_radius(i0, j0, r, spin_lattice, T):
    i = i0 + r
    j = [j0]
    no_change = True
    while i > i0 - r - 1:
        for j_c in j:
            s = switch(i, j_c, spin_lattice, T)
            no_change = no_change and spin_lattice[i, j_c] == s
            spin_lattice[i, j_c] = s
        i -= 1
        j = calculate_second_index(i0, j0, i, r)
        if j[0] == j[1]:
            j.pop()

    return spin_lattice, no_change


def most_probable(T, size, iter=1000, R=5, graph=True, start_lattice=None):
    spin_lattice = np.pad(np.random.randint(2, size=size)*2 - 1, 1, mode='constant') if start_lattice is None else start_lattice
    avs = 0
    ac = 0

    if graph:
        im = plt.imshow(spin_lattice)
        plt.show(block=False)
        plt.pause(0.5)

    for k in range(iter):
        i0 = np.random.randint(1, size[0]+1)
        j0 = np.random.randint(1, size[1]+1)

        s = switch(i0, j0, spin_lattice, T)

        if spin_lattice[i0, j0] != s:
            spin_lattice[i0, j0] = s
            for r in range(1, R):
                if calculate_radius(i0, j0, 1, j0) < r or calculate_radius(i0, j0, i0, 1) < r or calculate_radius(i0, j0, size[0], j0) < r or calculate_radius(i0, j0, i0, size[1]) < r:
                    break
                else:
                    spin_lattice, end = switch_radius(i0, j0, r, spin_lattice, T)
                    if end:
                        break

        if k >= 0.99*iter:
            avs += calculate_average_spin(spin_lattice)
            ac += 1
        if graph:
            if k % round(1000/R) == 0:
                plt.pause(0.000001)
                im.set_data(spin_lattice)
                plt.draw()
    if graph:
        plt.pause(1)
        plt.close()

    return avs/ac if ac > 5 else calculate_average_spin(spin_lattice), calculate_average_spin(spin_lattice), count_energy(spin_lattice), spin_lattice


if __name__ == '__main__':
    size = np.array([50, 50])

    temperatures = np.linspace(0.00001, 10)
    start_lattice = most_probable(0.00001, size, iter=2000000, R=5, graph=False)[3]
    plt.imshow(start_lattice)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    spins = list()
    for t in tq.tqdm(temperatures):
        mp = most_probable(t, size, iter=50000, R=3, graph=False, start_lattice=start_lattice)
        start_lattice = mp[3]
        spins.append(mp[0])
    plt.plot(temperatures, spins)
    plt.show()
