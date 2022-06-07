import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def PLOT_BAR(OPT, STD, grace, RGB, model_name=" "):
    figure(figsize=(10, 8), dpi=1024)
    data = np.array([OPT, STD, grace, RGB])
    data = data.T
    columns = ('KLT', 'Grace', 'STD', 'RGB')
    rows = ['S/Y/R', 'W/U/G', 'X/V/B']
    values = np.arange(0, 500, 100) / 500.
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
    plt.ylabel("L2 Sensitivity(normalized)")
    plt.yticks(values, ['%1.1f' % val for val in values])
    plt.xticks([])
    plt.title(model_name)
    plt.savefig(model_name+".pdf")

Alexnet_OPT = [0.621704,   0.22092277, 0.15737323]
Alexnet_STD = [0.60630638, 0.20284967, 0.19084395]
Alexnet_grace = [0.58524348, 0.1829203,  0.23183622]
Alexnet_RGB = [0.32434733, 0.45129792, 0.22435474]
PLOT_BAR(Alexnet_OPT, Alexnet_STD, Alexnet_grace, Alexnet_RGB, "Alexnet")

VGG11_OPT = [0.76132283, 0.14542937, 0.0932478]
VGG11_STD = [0.73370767, 0.14627937, 0.12001296]
VGG11_grace = [0.7424967,  0.11916233, 0.13834098]
VGG11_RGB = [0.30519189, 0.50915815, 0.18564996]
PLOT_BAR(VGG11_OPT, VGG11_STD, VGG11_grace, VGG11_RGB, "VGG11")

Squeezenet_OPT = [0.80399823, 0.11221068, 0.08379109]
Squeezenet_STD = [0.76868306, 0.12688239, 0.10443455]
Squeezenet_grace = [0.77997787, 0.10171819, 0.11830394]
Squeezenet_RGB = [0.30147736, 0.5271912,  0.17133144]
PLOT_BAR(Squeezenet_OPT, Squeezenet_STD, Squeezenet_grace, Squeezenet_RGB, "Squeezenet")

Resnet18_OPT = [0.79763572, 0.1238097,  0.07855458]
Resnet18_STD = [0.76388099, 0.13016013, 0.10595888]
Resnet18_grace = [0.78420441, 0.09597062, 0.11982497]
Resnet18_RGB = [0.30665523, 0.51333355, 0.18001122]
PLOT_BAR(Resnet18_OPT, Resnet18_STD, Resnet18_grace, Resnet18_RGB, "Resnet18")
