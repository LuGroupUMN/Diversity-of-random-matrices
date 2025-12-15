import torch
import matplotlib.pyplot as plt
from datetime import datetime

mdl = torch.load("trained_model", weights_only=False)

P = mdl.state_dict()["P"].cpu().numpy()
Q = mdl.state_dict()["Q"].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

im1 = axes[0].imshow(P, cmap="viridis")
axes[0].set_title("P")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(Q, cmap="viridis")
axes[1].set_title("Q")
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(P @ Q, cmap="viridis")
axes[2].set_title("P @ Q")
plt.colorbar(im3, ax=axes[2])

fig.suptitle(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
print(f'Saved to heatmap.png')

print(P)
print(Q)
print(P @ Q)