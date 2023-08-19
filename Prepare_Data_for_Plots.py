import numpy as np

# set comma if columns not processed
with open('Adam_train_and_test_losses_500_transport_c1_19_set.csv', 'r') as file:
    lines = file.readlines()

# modify the rows
modified_lines = []
for line in lines:
    modified_line = line.replace(' ', ',', 50)  # Replace only (mozart: two)(or transport: 51) occurrences of space with comma
    modified_lines.append(modified_line)

# write the modified rows back to the CSV file
with open('comma_Adam_train_and_test_losses_500_transport_c1_19_set.csv', 'w') as file:
    file.writelines(modified_lines)

csv1 = np.genfromtxt('comma_Adam_train_and_test_losses_500_transport_c1_19_set.csv', delimiter=',', dtype=float)
csv2 = np.genfromtxt('Adam_test_data_0.001_500_transport_c1_19_set.csv', delimiter=',', dtype=float)
print(csv1.shape)
print(csv2.shape)

merged_data = np.hstack((csv1[:, [0]], csv2))

# save the merged data to a new CSV file
np.savetxt('merged_Adam_train_and_test_losses_500_transport_c1_19_set.csv', merged_data, delimiter=',', fmt='%s')
