import numpy as np
import matplotlib.pyplot as plt


data = np.array([[0.621704,   0.22092277, 0.15737323],
                [0.621704,   0.22092277, 0.15737323],
                [0.621704,   0.22092277, 0.15737323],
                [0.621704,   0.22092277, 0.15737323]])
data = data.T
columns = ('KLT', 'Grace', 'STD', 'RGB')
rows = ['S/Y/R', 'W/U/G', 'X/V/B']
values = np.arange(0, 500, 100)/500.
value_increment = 1
# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)
index = np.arange(len(columns)) + 0.3
bar_width = 0.5
# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))
# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.4f' % (x) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.ylabel("Loss in ${0}'s".format(value_increment))
plt.yticks(values , ['%1.1f' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')
plt.show()