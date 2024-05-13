import numpy as np
import matplotlib.pyplot as plt
import tqdm as tq

kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def calculate_average_spin(state):
    return np.abs(np.sum(state))/np.prod(state.shape)


def count_energy(state):
    h, w = state.shape
    energy = 0
    for i in range(1, h-1):
        for j in range(1, w-1):
            energy += np.sum(state[i, j]*kernel*state[i-1:i+2, j-1:j+2])
    return energy/2


def most_probable(T, size, iter=1000):
    spin_lattice = np.pad(np.random.randint(2, size=size)*2 - 1, 1, mode='constant')
    avs = 0
    ac = 0

    im = plt.imshow(spin_lattice)
    plt.show(block=False)
    plt.pause(0.5)

    for k in tq.tqdm(range(iter)):
        spin_lattice_clone = np.copy(spin_lattice)
        for i in range(1, size[0]-1):
            for j in range(1, size[1]-1):
                E = np.sum(spin_lattice[i, j]*kernel*spin_lattice[i-1:i+2, j-1:j+2])
                nE = np.sum((-spin_lattice[i, j])*kernel*spin_lattice[i-1:i+2, j-1:j+2])
                if nE > E:
                    spin_lattice_clone[i, j] = -spin_lattice[i, j]
                else:
                    p = np.exp((nE-E)/T) if T != 0 else 0
                    if np.random.choice([0, 1], 1, p=[1-p, p]) == 1:
                        spin_lattice_clone[i, j] = -spin_lattice[i, j]
        spin_lattice = spin_lattice_clone
        if k >= 0.5*iter:
            avs += calculate_average_spin(spin_lattice)
            ac += 1
        if k % 100 == 0:
            plt.pause(0.000001)
            im.set_data(spin_lattice)
            plt.draw()
    plt.pause(1)
    plt.close()

    return avs/ac if ac > 20 else calculate_average_spin(spin_lattice), calculate_average_spin(spin_lattice), count_energy(spin_lattice)


if __name__ == '__main__':
    size = np.array([100, 100])

    temperatures = np.linspace(0, 10)
    plt.plot(temperatures, [most_probable(t, size, iter=10000)[0] for t in tq.tqdm(temperatures)])
    plt.show()
