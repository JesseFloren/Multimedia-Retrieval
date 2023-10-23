import numpy as np
import random


### -- Setup -- ###
def create_vector(n, range_min, range_max):
    vec = np.array([random.uniform(range_min, range_max) for _ in range(n)])
    return vec

def create_perturbed_vector(vec, range_min, range_max, perturbation_size):
    result = []
    for number in vec:
        # Perturb item in vector by random amount (within predefined range)
        number += np.random.uniform(-perturbation_size, perturbation_size)

        # Enforce range [range_min, range_max]
        number = min(range_max, number)
        number = max(range_min, number)

        result.append(number)

    return np.array(result)


### -- Distance functions -- ###
def get_manhattan_distance(vec_a, vec_b, range_min, range_max, normalize=True):
    dist = 0
    for number_a, number_b in zip(vec_a, vec_b):
        dist += abs(number_a - number_b)
    if normalize:
        max_dist = (range_max - range_min) * len(vec_a)
        dist /= max_dist

    return dist

def get_euclidean_distance(vec_a, vec_b, range_min, range_max, normalize=True):
    dist = np.linalg.norm(vec_a - vec_b)
    if normalize:
        max_dist = np.sqrt(len(vec_a) * ((range_max - range_min)**2))
        dist /= max_dist
    
    return dist

def get_cosine_distance(vec_a, vec_b, normalize=True):
    cosine_similarity = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))    
    dist = 1 - cosine_similarity
    if normalize:
        dist /= 2
    
    return dist


def main():
    # Parameters
    n = 3
    range_min = -5
    range_max = 5
    perturbation_size = 10

    # Create a random vector and a perturbed copy
    vec_a = create_vector(n, range_min, range_max)
    vec_b = create_perturbed_vector(vec_a, range_min, range_max, perturbation_size)

    #vec_a = np.array([1]*n)
    #vec_b = np.array([1]*int(n/2) + [-1]*int(n/2))

    # Results
    print("Vector a:", vec_a)
    print("Vector b:", vec_b)
    print("Euclidean distance:", get_euclidean_distance(vec_a, vec_b, range_min, range_max))
    print("Cosine distance:", get_cosine_distance(vec_a, vec_b))
    print("Manhattan distance:", get_manhattan_distance(vec_a, vec_b, range_min, range_max))
    if range_min > 0:
        import emd
        print("Earth Mover's Distance:", emd.get_emd(vec_a, vec_b))

    # Visualize the feature vectors using a plot (if #dimensions is equal to 2 or 3)
    if n == 2:
        from matplotlib import pyplot as plt
        vectors = np.array([vec_a, vec_b])
        origin = np.array([[0, 0],[0, 0]])
        plt.quiver(*origin, vectors[:,0], vectors[:,1], color=['r','b'], scale=(range_max - range_min)*1.5)
        plt.show()
    elif n == 3:
        import matplotlib.pyplot as plt

        soa = np.array([[0]*3 + [item for item in vec_a], [0]*3 + [item for item in vec_b]])
        X, Y, Z, U, V, W = zip(*soa)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, U, V, W)
        ax.set_xlim([range_min, range_max])
        ax.set_ylim([range_min, range_max])
        ax.set_zlim([range_min, range_max])
        plt.show()


if __name__ == "__main__":
    main()

