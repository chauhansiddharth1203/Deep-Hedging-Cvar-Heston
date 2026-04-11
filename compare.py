import torch
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results/plots", exist_ok=True)

pnl_var = torch.load("results/pnl_variance_deep.pt").cpu().numpy()
pnl_cvar = torch.load("results/pnl_deep.pt").cpu().numpy()
pnl_delta = torch.load("results/pnl_variance_delta.pt").cpu().numpy()

plt.figure(figsize=(8, 5))
plt.hist(pnl_var, bins=100, density=True, alpha=0.5, label="Deep Hedge (Variance)")
plt.hist(pnl_cvar, bins=100, density=True, alpha=0.5, label="Deep Hedge (CVaR)")
plt.hist(pnl_delta, bins=100, density=True, alpha=0.5, label="Delta Hedge")
plt.xlabel("PnL")
plt.ylabel("Density")
plt.legend()
plt.title("PnL Distribution Across Risk Objectives")
plt.tight_layout()
plt.savefig("results/plots/pnl_overlay_all.png", dpi=300)
plt.close()

print("Objective comparison plot generated.")
