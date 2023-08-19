import numpy as np
import matplotlib.pyplot as plt

# load the saved R2 Scores and Losses for training and test data for the example of the transport problem of the case with p ∈ R^(12) 

# file = {}
# for i in range(18)
#   file[i] = np.loadtxt(f'Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_{i+1}_0.001_set.txt', delimiter=',')

# easier to plot only certain files, 9 - 11 are intentionally missing, as these no. were not considered for this case
# R2 Scores, for Losses use corresponding files and decomment l. 83f.
file0 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_1_0.001_set.txt', delimiter=',')
file12 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_13_0.001_set.txt', delimiter=',')
file13 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_14_0.001_set.txt', delimiter=',')
file14 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_15_0.001_set.txt', delimiter=',')
file2 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_3_0.001_set.txt', delimiter=',')
file3 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_4_0.001_set.txt', delimiter=',')
file4 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_5_0.001_set.txt', delimiter=',')
file5 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_6_0.001_set.txt', delimiter=',')
file6 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_7_0.001_set.txt', delimiter=',')
file15 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_16_0.001_set.txt', delimiter=',')
file16 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_17_0.001_set.txt', delimiter=',')
file7 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_8_0.001_set.txt', delimiter=',')
file8 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_9_0.001_set.txt', delimiter=',')
file17 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_18_0.001_set.txt', delimiter=',')
file1 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_2_0.001_set.txt', delimiter=',')
file18 = np.loadtxt('Adam_R2_train_and_test_scores_500_transport_b1-b12;b22-30_19_0.001_set.txt', delimiter=',')

# create figure and subplots
fig, ax = plt.subplots(2,1,figsize=(8, 8))

# set the colormap
colors = plt.cm.tab20.colors

# plot data
ax[0].plot(file0[:, 0], label='[12,2,50]', color=colors[1])
ax[1].plot(file0[:, 1], label='[12,2,50]', color=colors[1])
ax[0].plot(file12[:, 0], label='[12,2,2,50]', color=colors[13])
ax[1].plot(file12[:, 1], label='[12,2,2,50]', color=colors[13])
ax[0].plot(file13[:, 0], label='[12,2,2,2,50]', color=colors[14])
ax[1].plot(file13[:, 1], label='[12,2,2,2,50]', color=colors[14])
ax[0].plot(file14[:, 0], label='[12,2,5,50]', color=colors[15])
ax[1].plot(file14[:, 1], label='[12,2,5,50]', color=colors[15])
ax[0].plot(file2[:, 0], label='[12,5,50]', color=colors[3])
ax[1].plot(file2[:, 1], label='[12,5,50]', color=colors[3])
ax[0].plot(file3[:, 0], label='[12,10,50]', color=colors[4])
ax[1].plot(file3[:, 1], label='[12,10,50]', color=colors[4])
ax[0].plot(file4[:, 0], label='[12,15,50]', color=colors[5])
ax[1].plot(file4[:, 1], label='[12,15,50]', color=colors[5])
ax[0].plot(file5[:, 0], label='[12,20,50]', color=colors[6])
ax[1].plot(file5[:, 1], label='[12,20,50]', color=colors[6])
ax[0].plot(file6[:, 0], label='[12,25,50]', color=colors[7])
ax[1].plot(file6[:, 1], label='[12,25,50]', color=colors[7])
ax[0].plot(file15[:, 0], label='[12,5,5,50]', color=colors[16])
ax[1].plot(file15[:, 1], label='[12,5,5,50]', color=colors[16])
ax[0].plot(file16[:, 0], label='[12,2,10,50]', color=colors[17])
ax[1].plot(file16[:, 1], label='[12,2,10,50]', color=colors[17])
ax[0].plot(file7[:, 0], label='[12,30,50]', color=colors[8])
ax[1].plot(file7[:, 1], label='[12,30,50]', color=colors[8])
ax[0].plot(file8[:, 0], label='[12,40,50]', color=colors[9])
ax[1].plot(file8[:, 1], label='[12,40,50]', color=colors[9])
ax[0].plot(file17[:, 0], label='[12,5,10,50]', color=colors[18])
ax[1].plot(file17[:, 1], label='[12,5,10,50]', color=colors[18])
ax[0].plot(file1[:, 0], label='[12,62,50]', color=colors[2])
ax[1].plot(file1[:, 1], label='[12,62,50]', color=colors[2])
ax[0].plot(file18[:, 0], label='[12,10,10,50]', color=colors[19])
ax[1].plot(file18[:, 1], label='[12,10,10,50]', color=colors[19])

# Add labels and legend
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Training Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Test Accuracy')

# Add legend
ax[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), borderaxespad=0, fontsize='small')
ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), borderaxespad=0, fontsize='small')

# Add title
ax[0].set_title("Training")
ax[1].set_title("Test")

#ax[0].set_yscale('log')
#ax[1].set_yscale('log')

plt.tight_layout()

plt.savefig('plot.eps', format='eps')

# Show plot
plt.show()
