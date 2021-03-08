import csv
import numpy as np
import matplotlib.pyplot as plt

error_list = []
with open('error_statistics.csv', 'r', newline='', encoding='utf-8') as file:
    r = csv.reader(file)
    for e in r:
        error_list.append(float(e[0]))
print('Mean L2-error: ', np.mean(error_list))
print('Standard deviation L2-error: ', np.std(error_list))

plt.boxplot(error_list)
plt.show()




