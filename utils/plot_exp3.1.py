import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.style'] = 'normal'



# list all folder names in the directory
abs_dir = os.path.dirname(os.getcwd())
source_dir = os.path.join(abs_dir, 'model/exp3-transfer/transfer_results_one2multi')

loss_table = np.load(os.path.join(source_dir, 'test_loss.npy'))

loss_table = np.round(loss_table, decimals=5)

# create the table
table_data = []
for i, loss in enumerate(loss_table[4:11,4:11]):
    table_data.append(['Training on {} cables'.format(i+4)] + loss.tolist())

headers = ["", "Eval on 4 cables", "Eval on 5 cables", "Eval on 6 cables", "Eval on 7 cables", "Eval on 8 cables",
           "Eval on 9 cables", "Eval on 10 cables"]

# Create the table using matplotlib
fig, ax = plt.subplots()
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# Save the table as a PNG image
plt.show()
# plt.savefig(os.path.join(source_dir, 'table.png'))
plt.close()