import matplotlib.pyplot as plt
import numpy as np

# Replace these random numbers with your actual validation losses
val_loss_7_7 = round(np.load('model/model_7cables/val_loss.npy')[-1], 4)
val_loss_7_9 = 2.1540


val_loss_7_7_ = 0.0051
val_loss_7_9_ = 0.0481

val_loss_9_7 = 0.2374
val_loss_9_9 = round(np.load('model/model_9cables/val_loss.npy')[-1], 4)

val_loss_9_7_ = 0.1129
val_loss_9_9_ = 0.0023

table_data = [
    ["Training on 7 cables", val_loss_7_7, val_loss_7_9],
    ["Training on 7 cables(fine-tune)", val_loss_7_7_, val_loss_7_9_],
    ["Training on 9 cables", val_loss_9_7, val_loss_9_9],
    ["Training on 9 cables(fine-tune)", val_loss_9_7_, val_loss_9_9_],
]

headers = ["", "Test on 7 cables", "Test on 9 cables"]

# Create the table using matplotlib
fig, ax = plt.subplots()
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# Save the table as a PNG image
plt.show()
plt.savefig("fig/table.png",bbox_inches="tight")