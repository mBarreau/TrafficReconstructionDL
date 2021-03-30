import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_data(filename):
    error = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        r = csv.reader(file)
        for e in r:
            error.append(float(e[0]))

    print('Mean:', np.mean(error))
    print('Standard deviation:', np.std(error))
    return error

print('\nL2 error L4DC:')
error1 = get_data('example/error_L4DC.csv')
print('\nL2 error BFGS:')
error2 = get_data('example/error_BFGS.csv')
print('\nComputation time L4DC:')
time1 = get_data('example/computation_time_L4DC.csv')
print('\nComputation time BFGS:')
time2 = get_data('example/computation_time_BFGS.csv')

figError = plt.figure()
plt.grid('on')
data = pd.DataFrame(np.array([error1, error2]).T, columns=["Modified Training", "Naive BFGS"])
sns.boxplot(data=data)
plt.ylabel('Normalized L2-error')
plt.tight_layout()

figTime = plt.figure()
plt.grid('on')
data = pd.DataFrame(np.array([time1, time2]).T, columns=["Modified Training", "Naive BFGS"])
sns.boxplot(data=data)
plt.ylabel('Computation time [s]')
plt.tight_layout()

figError.savefig('example/boxplot_error.eps', bbox_inches='tight')
figTime.savefig('example/boxplot_time.eps', bbox_inches='tight')

plt.show()




