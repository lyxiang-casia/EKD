# import matplotlib.pyplot as plt

# # Sample data for demonstration purposes
# T_values = range(7)  # Values for T on the x-axis
# J_M_scores = [98.68, 95.33, 94.39, 93.89, 93.12, 92.23, 91.04]  # Scores for J_M
# F_M_scores = [73.76, 76.59, 77.11, 77.21, 77.13, 76.79, 76.39]  # Scores for F_M

# # Plotting
# plt.figure(figsize=(8, 4))
# plt.plot(T_values, J_M_scores, 'r^-', label=r'val', linestyle='--')
# plt.plot(T_values, F_M_scores, 'bo-', label=r'train', linestyle='--')

# # Adding data labels
# for i, score in enumerate(J_M_scores):
#     plt.text(T_values[i], J_M_scores[i], f"{score:.1f}", ha='right', va='bottom', color='red')
# for i, score in enumerate(F_M_scores):
#     plt.text(T_values[i], F_M_scores[i], f"{score:.1f}", ha='left', va='bottom', color='blue')

# # Labels and title
# plt.xlabel(r'$\mathcal{L}_{1st}:\mathcal{L}_{2nd}$')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True, linestyle=':', linewidth=0.5)

# # Save the figure
# plt.savefig("/data/mmc_lyxiang/KD/logit-standardization-KD-master/plot_with_values.png", format="png", dpi=300)  # Save as PNG with high resolution

# plt.show()

# import matplotlib.pyplot as plt

# # Custom x-axis labels
# x_labels = ['4.5:0', '4.5:1.5', '4.5:3', '4.5:4.5', '3:4.5', '1.5:4.5', '0:4.5']
# T_values = range(len(x_labels))  # Indices for T on the x-axis

# # Sample data
# J_M_scores_1 = [98.68, 95.33, 94.39, 93.89, 93.12, 92.23, 91.04]  # Scores for J_M
# F_M_scores_1 = [73.76, 76.59, 77.11, 77.21, 77.13, 76.79, 76.39]  # Scores for F_M

# J_M_scores_2 = [96.84, 94.46, 93.87, 93.71, 92.89, 92.12, 91.01]  # Scores for J_M
# F_M_scores_2 = [72.81, 76.05, 76.35, 76.51, 76.45, 76.23, 76.03]  # Scores for F_M

# # Plotting
# plt.figure(figsize=(8, 4))
# plt.plot(T_values, J_M_scores, 'r^-', label=r'val', linestyle='--')
# plt.plot(T_values, F_M_scores, 'bo-', label=r'train', linestyle='--')

# # Adding data labels
# for i, score in enumerate(J_M_scores):
#     plt.text(T_values[i], J_M_scores[i], f"{score:.1f}", ha='right', va='bottom', color='red')
# for i, score in enumerate(F_M_scores):
#     plt.text(T_values[i], F_M_scores[i], f"{score:.1f}", ha='left', va='bottom', color='blue')

# # Setting custom x-axis labels
# plt.xticks(T_values, x_labels)

# # Labels and legend
# plt.xlabel(r'$\mathcal{L}_{1st}:\mathcal{L}_{2nd}$')
# plt.ylabel('Accuracy')
# plt.legend(frameon=False, loc='best')
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_color('black')
# plt.gca().spines['left'].set_color('black')
# plt.annotate('', xy=(1, 0), xytext=(-0.1, 0),
#              xycoords='axes fraction', textcoords='axes fraction',
#              arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

# plt.annotate('', xy=(0, 1), xytext=(0, -0.1),
#              xycoords='axes fraction', textcoords='axes fraction',
#              arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
# plt.grid(False)

# # Save the figure
# plt.savefig("/data/mmc_lyxiang/KD/logit-standardization-KD-master/plot_with_values.png", format="png", dpi=300, transparent=True, bbox_inches='tight')  # Save as PNG with high resolution

# plt.show()
import matplotlib.pyplot as plt

# Sample data for two different sets
x_labels = ['1:0', '1:0.25', '1:0.5', '1:1', '0.5:1', '0.25:1', '0:1']
T_values = range(7)  # Values for T on the x-axis
# J_M_scores_1 = [98.68, 95.33, 94.39, 93.89, 93.12, 92.23, 91.04]  # Scores for J_M (set 1)
F_M_scores_1 = [73.76, 76.59, 77.11, 77.21, 77.13, 76.79, 76.39]  # Scores for F_M (set 1)

# J_M_scores_2 = [96.84, 94.46, 93.87, 93.71, 92.89, 92.12, 91.01]  # Scores for J_M (set 2)
F_M_scores_2 = [72.81, 76.05, 76.35, 76.51, 76.45, 76.23, 76.03]  # Scores for F_M (set 2)

# Plotting
plt.figure(figsize=(8, 4))

# Plot the _1 series in red and _2 series in green
# plt.plot(T_values, J_M_scores_1, 'r^-', label=r'ResNet8x4 (val)', linestyle='--')
plt.plot(T_values, F_M_scores_1, 'ro-', label=r'ResNet8x4', linestyle='--')
# plt.plot(T_values, J_M_scores_2, 'b^-', label=r'WRN-16-2 (val)', linestyle='--')
plt.plot(T_values, F_M_scores_2, 'bo-', label=r'WRN-16-2', linestyle='--')

# Adding data labels
# for i, score in enumerate(J_M_scores_1):
#     plt.text(T_values[i], J_M_scores_1[i] + 2.0, f"{score:.1f}", ha='center', va='top', color='red')  # Above the line
for i, score in enumerate(F_M_scores_1):
    plt.text(T_values[i], F_M_scores_1[i] + 1.0, f"{score:.1f}", ha='center', va='top', color='red')  # Below the line

# for i, score in enumerate(J_M_scores_2):
#     plt.text(T_values[i], J_M_scores_2[i] - 2.0, f"{score:.1f}", ha='center', va='bottom', color='blue')  # Above the line
for i, score in enumerate(F_M_scores_2):
    plt.text(T_values[i], F_M_scores_2[i] - 1.0, f"{score:.1f}", ha='center', va='bottom', color='blue')  # Below the line

# Labels and title
# plt.xlabel(r'$\mathcal{L}_{1st}:\mathcal{L}_{2nd}$')
# plt.ylabel('Accuracy')
# plt.legend()

plt.xticks(T_values, x_labels)
plt.xlabel(r'$\mathcal{L}_{1st}:\mathcal{L}_{2nd}$')
plt.ylabel('Accuracy')
plt.ylim(70, 80)

plt.axhline(y=73.33, color='red', linestyle=':', linewidth=1.5, label='ResNet8x4 Baseline (KD)')
plt.axhline(y=74.90, color='blue', linestyle=':', linewidth=1.5, label='WRN-16-2 Baseline (KD)')

# 可选注释文字
# plt.text(6.2, 73.33 + 0.2, 'Baseline: 73.33', color='red', fontsize=9)
# plt.text(6.2, 74.90 + 0.2, 'Baseline: 74.90', color='blue', fontsize=9)



# 添加图例，设置边框
plt.legend(frameon=True, edgecolor='black', bbox_to_anchor=(0.9, 0.33))

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.grid(False)

# Save the figure
plt.savefig("/data/mmc_lyxiang/KD/logit-standardization-KD-master/plot_with_values.png", format="png", dpi=300, bbox_inches='tight', transparent=True)

# Show the plot
plt.show()



