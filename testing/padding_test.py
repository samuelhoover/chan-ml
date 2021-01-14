import time
import numpy as np

# Testing speed of inserting `a` into `b` versus np.pad

# RESULTS:
#   1.) For both single arrays and lists, np.pad is slightly quicker (~5 % quicker) than the insertion method


def stats(arr_1, time_1, arr_2, time_2):
    # Evaluate runtime and calculate speedup
    if time_1 < time_2:
        speedup = time_2 / time_1
    
    else:
        speedup = time_1 / time_2

    # Check for identical results between the two padding methods
    # for list_test()
    if type(arr_1) is list:
        for idx, _ in enumerate(arr_1):
            print('Identical results?: {}'.format(np.array_equal(arr_1[idx], arr_2[idx])))
            if not np.array_equal(arr_1[idx], arr_2[idx]):
                print('Difference: {}'.format(np.abs(np.sum(arr_1[idx] - arr_2[idx]))))
    
    # for single_array_test()
    else:
        print('Identical results?: {}'.format(np.array_equal(arr_1, arr_2)))
        if not np.array_equal(arr_1, arr_2):
            print('Difference: {}'.format(np.abs(np.sum(arr_1 - arr_2))))

    print('Insertion method: {} s'.format(time_1))
    print('Padding method: {} s'.format(time_2))
    print('Speed up: {}'.format(speedup))


def single_array_test():
    # Want to pad `a` with zeros such that it becomes the size of `b`
    print('Single array test ...')
    a = np.random.random((100, 60, 80))
    b = np.zeros((550, 286, 210))

    num_iter = 50
    runtime_ins = 0
    runtime_pad = 0
    for i in range(num_iter):
        # Method 1: insertion
        t0 = time.time()
        result_ins = np.copy(b)
        result_ins[:a.shape[0], :a.shape[1], :a.shape[2]] = a
        t1 = time.time()

        # Method 2: np.pad
        t2 = time.time()
        size_diff = np.asarray(b.shape) - np.asarray(a.shape)
        result_pad = np.pad(a, [(0, size_diff[0]), (0, size_diff[1]), (0, size_diff[2])], mode='constant')
        t3 = time.time()

        runtime_ins += (t1 - t0)
        runtime_pad += (t3 - t2)

    runtime_ins /= num_iter
    runtime_pad /= num_iter
    stats(result_ins, runtime_ins, result_pad, runtime_pad)


def list_test():
    # Want to pad each element in `a` with zeros such that it becomes the size of `b`
    print('List test ...')
    a = [
        np.random.random((100, 60, 80)),
        np.random.random((70, 60, 60)),
        np.random.random((25, 140, 140)),
        np.random.random((70, 110, 130)),
        np.random.random((50, 120, 70)),
    ]
    
    b = np.zeros((550, 286, 210))
    result_ins = []
    result_pad = []

    num_iter = 20
    runtime_ins = 0
    runtime_pad = 0
    for iter in range(num_iter):
        # Method 1: insertion
        t0 = time.time()
        for i, _ in enumerate(a):
            intrm = np.copy(b)
            intrm[:a[i].shape[0], :a[i].shape[1], :a[i].shape[2]] = a[i]
            result_ins.append(intrm)
       
        t1 = time.time()

        # Method 2: np.pad
        t2 = time.time()
        for j, _ in enumerate(a):
            size_diff = np.asarray(b.shape) - np.asarray(a[j].shape)
            result_pad.append(np.pad(a[j], [(0, size_diff[0]), (0, size_diff[1]), (0, size_diff[2])], mode='constant'))
        
        t3 = time.time()

        runtime_ins += (t1 - t0)
        runtime_pad += (t3 - t2)

    runtime_ins /= num_iter
    runtime_pad /= num_iter
    stats(result_ins, runtime_ins, result_pad, runtime_pad)


def main():
    single_array_test()


if __name__ == "__main__":
    main()
