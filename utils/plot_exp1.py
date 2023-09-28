import matplotlib.pyplot as plt
import numpy as np
import os


# list all folder names in the directory
abs_dir = os.path.dirname(os.getcwd())
source_dir = os.path.join(abs_dir, 'model/exp1-cfgs')
# folder_names = os.listdir(source_dir)
folder_names = [item for item in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, item))]

#sort the folder names
folder_names.sort()

loss_table = {}
for folder in folder_names:
    # load the test loss
    loss_list = []
    current_dir_name = os.path.join(source_dir, folder)
    for i in range(4, 11):
        _loss = np.load(os.path.join(current_dir_name,'model_{}cables'.format(i), 'val_loss.npy'))
        if _loss.shape != ():
            loss_list.append(_loss[-1])
        else:
            loss_list.append(_loss)
        # loss_list.append(np.load(os.path.join(current_dir_name,'model_{}cables'.format(i), 'val_loss.npy'))[-1])
    loss_table[folder] = loss_list



# create the table
table_data = []
for key, value in loss_table.items():
    table_data.append([key] + value)

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