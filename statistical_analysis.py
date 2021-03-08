import csv
import numpy as np
import matplotlib.pyplot as plt

def get_error(filename):
    error = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        r = csv.reader(file)
        for e in r:
            error.append(float(e[0]))

    print('Mean L2-error: ', np.mean(error))
    print('Standard deviation L2-error: ', np.std(error))
    return error

print('\nL4DC algorithm:')
error1 = get_error('error_L4DC.csv')
print('\nEqual loss weights, BFGS:')
error2 = get_error('error_BFGS.csv')

fig = plt.figure()
data = [error1, error2]
plt.boxplot(data)
plt.xticks([1, 2], ['L4DC algorithm', 'Equal loss weights, BFGS'])

fig.savefig('boxplot.eps', bbox_inches='tight')
plt.show()




