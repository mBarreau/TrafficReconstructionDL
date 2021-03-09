import csv
import numpy as np
import matplotlib.pyplot as plt

def get_data(filename):
    error = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        r = csv.reader(file)
        for e in r:
            error.append(float(e[0]))

    print('Mean: ', np.mean(error))
    print('Standard deviation: ', np.std(error))
    return error

print('\nL2 error L4DC:')
error1 = get_data('error_L4DC.csv')
print('\nL2 error BFGS:')
error2 = get_data('error_BFGS.csv')
print('\nComputation time L4DC:')
print('\nComputation time BFGS:')
time2 = get_data('computation_time_BFGS.csv')

figError = plt.figure()
data = [error1, error2]
plt.ylabel('Normalized L2-error')
plt.boxplot(data)
plt.xticks([1, 2], ['L4DC algorithm', 'Equal loss weights, BFGS'])

figTime = plt.figure()
data = [[], time2]
plt.boxplot(data)
plt.ylabel('Computation time [s]')
plt.xticks([1, 2], ['L4DC algorithm', 'Equal loss weights, BFGS'])

figError.savefig('boxplot.eps', bbox_inches='tight')
figTime.savefig('computation_time.eps', bbox_inches='tight')

plt.figure()
plt.plot(time2)
plt.show()




