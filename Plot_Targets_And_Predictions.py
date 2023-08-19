import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

dataset = pd.read_csv('merged_Adam_train_and_test_losses_500_transport_c1_19_set.csv')

# scatter plot for transport problem, for mozart problem analogously 
plt.figure()
l1 = plt.scatter(dataset['p'], dataset['x1'], color='red', s=5) 
l2 = plt.scatter(dataset['p'], dataset['x2'], color='blue', s=5)
l3 = plt.scatter(dataset['p'], dataset['x3'], color='black', s=5)
l4 = plt.scatter(dataset['p'], dataset['x4'], color='darkgray', s=5)
l5 = plt.scatter(dataset['p'], dataset['x5'], color='indianred', s=5)
l6 = plt.scatter(dataset['p'], dataset['x6'], color='maroon', s=5)
l7 = plt.scatter(dataset['p'], dataset['x7'], color='lightsalmon', s=5)
l8 = plt.scatter(dataset['p'], dataset['x8'], color='peachpuff', s=5)
l9 = plt.scatter(dataset['p'], dataset['x9'], color='orange', s=5)
l10 = plt.scatter(dataset['p'], dataset['x10'], color='goldenrod', s=5)
l11 = plt.scatter(dataset['p'], dataset['x11'], color='gold', s=5)
l12 = plt.scatter(dataset['p'], dataset['x12'], color='darkkhaki', s=5)
l13 = plt.scatter(dataset['p'], dataset['x13'], color='olive', s=5)
l14 = plt.scatter(dataset['p'], dataset['x14'], color='yellow', s=5)
l15 = plt.scatter(dataset['p'], dataset['x15'], color='darkturquoise', s=5) #andere Farbe, zu nah an turquoises
l16 = plt.scatter(dataset['p'], dataset['x16'], color='yellowgreen', s=5)
l17 = plt.scatter(dataset['p'], dataset['x17'], color='darkolivegreen', s=5)
l18 = plt.scatter(dataset['p'], dataset['x18'], color='darkseagreen', s=5)
l19 = plt.scatter(dataset['p'], dataset['x19'], color='aquamarine', s=5)
l20 = plt.scatter(dataset['p'], dataset['x20'], color='limegreen', s=5)
l21 = plt.scatter(dataset['p'], dataset['x21'], color='lightseagreen', s=5)
l22 = plt.scatter(dataset['p'], dataset['x22'], color='orchid', s=5)
l23 = plt.scatter(dataset['p'], dataset['x23'], color='teal', s=5)
l24 = plt.scatter(dataset['p'], dataset['x24'], color='cadetblue', s=5)
l25 = plt.scatter(dataset['p'], dataset['x25'], color='deepskyblue', s=5)
l26 = plt.scatter(dataset['p'], dataset['x26'], color='steelblue', s=5)
l27 = plt.scatter(dataset['p'], dataset['x27'], color='royalblue', s=5)
l28 = plt.scatter(dataset['p'], dataset['x28'], color='navy', s=5)
l29 = plt.scatter(dataset['p'], dataset['x29'], color='slateblue', s=5)
l30 = plt.scatter(dataset['p'], dataset['x30'], color='blueviolet', s=5)
l31 = plt.scatter(dataset['p'], dataset['x31'], color='mediumorchid', s=5)
l32 = plt.scatter(dataset['p'], dataset['x32'], color='plum', s=5)
l33 = plt.scatter(dataset['p'], dataset['x33'], color='violet', s=5)
l34 = plt.scatter(dataset['p'], dataset['x34'], color='darkmagenta', s=5)
l35 = plt.scatter(dataset['p'], dataset['x35'], color='magenta', s=5)
l36 = plt.scatter(dataset['p'], dataset['x36'], color='turquoise', s=5)
l37 = plt.scatter(dataset['p'], dataset['x37'], color='hotpink', s=5)
l38 = plt.scatter(dataset['p'], dataset['x38'], color='crimson', s=5)
l39 = plt.scatter(dataset['p'], dataset['x39'], color='pink', s=5)
l40 = plt.scatter(dataset['p'], dataset['x40'], color='cadetblue', s=5)
l41 = plt.scatter(dataset['p'], dataset['x41'], color='orangered', s=5)
l42 = plt.scatter(dataset['p'], dataset['x42'], color='mediumaquamarine', s=5) #andere Farbe, zu nah an aquamarine
l43 = plt.scatter(dataset['p'], dataset['x43'], color='bisque', s=5)
l44 = plt.scatter(dataset['p'], dataset['x44'], color='purple', s=5)
l45 = plt.scatter(dataset['p'], dataset['x45'], color='thistle', s=5)
l46 = plt.scatter(dataset['p'], dataset['x46'], color='deeppink', s=5)
l47 = plt.scatter(dataset['p'], dataset['x47'], color='olivedrab', s=5)
l48 = plt.scatter(dataset['p'], dataset['x48'], color='slategray', s=5)
l49 = plt.scatter(dataset['p'], dataset['x49'], color='indigo', s=5)
l50 = plt.scatter(dataset['p'], dataset['x50'], color='cornflowerblue', s=5)

# Set the axis labels and title
plt.xlabel('Parameter p')
plt.ylabel('value of each component of minimizer')
plt.title('Scatter Plot')

# l. 66 sufficient for mozart problem
# plt.legend(loc='upper left', bbox_to_anchor=(0, 1), borderaxespad=0, fontsize='small')

legend1 = plt.legend((l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25), ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25"], loc = "lower right", bbox_to_anchor=(1.1, 0.1), borderaxespad=0, fontsize='small')
plt.gca().add_artist(legend1)
legend2 = plt.legend((l26, l27, l28, l29, l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42, l43, l44, l45, l46, l47, l48, l49, l50), ["x26", "x27", "x28", "x29", "x30", "x31", "x32", "x33", "x34", "x35", "x36", "x37", "x38", "x39", "x40", "x41", "x42", "x43", "x44", "x45", "x46", "x47", "x48", "x49", "x50"], loc = "lower right", bbox_to_anchor=(1.2, 0.1), borderaxespad=0, fontsize='small')
plt.gca().add_artist(legend2)

# Add a legend
plt.legend()

# Save the figure as EPS
plt.savefig('plot_merged_b1_5000_epochs_1500_2_predictions_0.001', format='eps')

# Show the plot
plt.show()
