import os
import json
import matplotlib.pyplot as plt


xp_folder = os.path.join('./', "xp_jamendo_full_5e-4")
fig_path = os.path.join(xp_folder, "losses.png")
data_path = os.path.join(xp_folder, "losses.json")

with open(data_path, 'r') as f:
    data = json.load(f)

# Extracting lists
train_loss = data["train_loss"]
test_loss = data["test_loss"]
epochs = range(1, len(train_loss) + 1)


# --- 2. PLOT INITIALIZATION ---
plt.figure(figsize=(10, 12))  # Taller aspect ratio matching your image
plt.plot(epochs, train_loss, label='Train Loss', color='blue')
plt.plot(epochs, test_loss, label='Test Loss', color='orange')


# --- 3. OVERFITTING DETECTION & ANNOTATION ---
overfit_epoch = None
# for i in range(1, len(test_loss)):
#     if test_loss[i] > test_loss[i-1]:
#         overfit_epoch = i + 1  # +1 for 1-indexed epochs
#         break

if overfit_epoch is not None:
    plt.annotate(
        'Overfitting Starts', 
        xy=(overfit_epoch, test_loss[overfit_epoch - 1]), 
        xytext=(overfit_epoch, test_loss[overfit_epoch - 1] + 0.2), 
        arrowprops=dict(facecolor='red', shrink=0.08, width=2, headwidth=8, headlength=8),
        fontsize=12,
        color='red',
        horizontalalignment='center',
        weight='medium'
    )
    plt.title('Train/Test Loss lr=5e-4 cond. drop=30%', fontsize=14, pad=15)
else:
    plt.title('Training and Testing Loss (No Overfitting Detected)', fontsize=14, pad=15)


# --- 4. CLEAN & DYNAMIC X-AXIS CONFIGURATION ---
ax = plt.gca()

# Generate clean base ticks every 10 epochs
tick_spacing = 5 
ticks = list(range(1, max(epochs) + 1, tick_spacing))

# Inject the exact overfitting epoch and clear out nearby collisions
if overfit_epoch is not None:
    ticks.append(overfit_epoch)
    # Removes any standard tick (like 20) if it sits too close to our target (like 19)
    ticks = [t for t in ticks if t == overfit_epoch or abs(t - overfit_epoch) > 2]

# Apply sorted, unique ticks to the axis
ticks = sorted(list(set(ticks)))
ax.set_xticks(ticks)

# Force matplotlib to render the ticks so we can style them
plt.draw() 

# Loop through labels and highlight the overfitting epoch baseline text
for label in ax.get_xticklabels():
    if label.get_text() == str(overfit_epoch):
        label.set_color('red')
        label.set_weight('bold')
        label.set_fontsize(12)


# --- 5. FINAL FORMATTING ---
plt.xlabel('Epochs', fontsize=12, labelpad=10)
plt.ylabel('Loss', fontsize=12, labelpad=10)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

# Display chart cleanly
plt.tight_layout()
plt.show()
plt.savefig(fig_path)