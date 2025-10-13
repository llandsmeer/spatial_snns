import numpy as np
import matplotlib.pyplot as plt
import h5py

samples = np.load("yin_yang_data_set/publication_data/train_samples.npy")
labels = np.load("yin_yang_data_set/publication_data/train_labels.npy")

timespan = 100
total_timespan = 500
scale = 0.5

times = []
units = []

for n in range(len(labels)):

    num_spikes_x = int((1-samples[n][0])*timespan) # np.random.poisson(lam = samples[n][0]*timespan*scale)
    num_spikes_y = int((1-samples[n][1])*timespan) #np.random.poisson(lam = samples[n][1]*timespan*scale)
    num_spikes_1_x = int((1-samples[n][2])*timespan) #np.random.poisson(lam = samples[n][2]*timespan*scale)
    num_spikes_1_y = int((1-samples[n][3])*timespan) #np.random.poisson(lam = samples[n][3]*timespan*scale)

    # if n == 100:
    #     print(samples[n])
    #     print(labels[n])
    #     print(n)

    dist_x = np.zeros(total_timespan, dtype = int) #np.concatenate((np.ones(num_spikes_x, dtype=int), np.zeros(timespan - num_spikes_x, dtype=int)))
    dist_y = np.zeros(total_timespan, dtype = int) #np.concatenate((np.ones(num_spikes_y, dtype=int), np.zeros(timespan - num_spikes_y, dtype=int)))
    dist_1_x = np.zeros(total_timespan, dtype = int) #np.concatenate((np.ones(num_spikes_1_x, dtype=int), np.zeros(timespan - num_spikes_1_x, dtype=int)))
    dist_1_y = np.zeros(total_timespan, dtype = int) #np.concatenate((np.ones(num_spikes_1_y, dtype=int), np.zeros(timespan - num_spikes_1_y, dtype=int)))
    dist_x[num_spikes_x] = 1
    dist_y[num_spikes_y] = 1
    dist_1_x[num_spikes_1_x] = 1
    dist_1_y[num_spikes_1_y] = 1


    times_t = []
    units_t = []
    for t in range(total_timespan):
        if dist_x[t] == 1:
            times_t.append(t)
            units_t.append(0)
        if dist_y[t] == 1:
            times_t.append(t)
            units_t.append(1)
        if dist_1_x[t] == 1:
            times_t.append(t)
            units_t.append(2)
        if dist_1_y[t] == 1:
            times_t.append(t)
            units_t.append(3)
    
    times.append(times_t)
    units.append(units_t)


times = np.array([np.array(xi) for xi in times])
units = np.array([np.array(xi) for xi in units])

print(times[0])
print(units[0])
print(len(times[0]))

plt.scatter(times[0], units[0])
plt.show()

# f = h5py.File("yy_rc_test.h5", 'w')
# times_dset = f.create_dataset("spikes/times", data=times)
# units_dset = f.create_dataset("spikes/units", data=units)
# labels_dset = f.create_dataset("labels", data=labels)