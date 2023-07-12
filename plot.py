import matplotlib.pyplot as plt
import numpy as np
import os

# Replace these random numbers with your actual validation losses
save_dir = 'model/transfer_results'
test_loss_list = np.load(os.path.join(save_dir, 'test_loss.npy'))
test_loss_list = np.round(test_loss_list, 4)

table_data = [
    ["Training on 4 cables", test_loss_list[4, 4], test_loss_list[4, 5], test_loss_list[4, 6], test_loss_list[4, 7],
     test_loss_list[4, 8], test_loss_list[4, 9], test_loss_list[4, 10]],

    ["Training on 5 cables", test_loss_list[5, 4], test_loss_list[5, 5], test_loss_list[5, 6], test_loss_list[5, 7],
     test_loss_list[5, 8], test_loss_list[5, 9], test_loss_list[5, 10]],

    ["Training on 6 cables", test_loss_list[6, 4], test_loss_list[6, 5], test_loss_list[6, 6], test_loss_list[6, 7],
     test_loss_list[6, 8], test_loss_list[6, 9], test_loss_list[6, 10]],

    ["Training on 7 cables", test_loss_list[7, 4], test_loss_list[7, 5], test_loss_list[7, 6], test_loss_list[7, 7],
     test_loss_list[7, 8], test_loss_list[7, 9], test_loss_list[7, 10]],

    ["Training on 8 cables", test_loss_list[8, 4], test_loss_list[8, 5], test_loss_list[8, 6], test_loss_list[8, 7],
     test_loss_list[8, 8], test_loss_list[8, 9], test_loss_list[8, 10]],

    ["Training on 9 cables", test_loss_list[9, 4], test_loss_list[9, 5], test_loss_list[9, 6], test_loss_list[9, 7],
     test_loss_list[9, 8], test_loss_list[9, 9], test_loss_list[9, 10]],

    ["Training on 10 cables", test_loss_list[10, 4], test_loss_list[10, 5], test_loss_list[10, 6],
     test_loss_list[10, 7], test_loss_list[10, 8], test_loss_list[10, 9], test_loss_list[10, 10]],
]

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
plt.savefig(os.path.join(save_dir, 'table.png'))
plt.close()