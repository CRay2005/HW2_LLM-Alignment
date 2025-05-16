import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open("reward_model_eval_results-new.json", "r", encoding="utf-8") as f:
    data = json.load(f)

scores_chosen = [d["score_chosen"] for d in data]
scores_rejected = [d["score_rejected"] for d in data]

# 绘制直方图
plt.figure(figsize=(10, 6))
sns.histplot(scores_chosen, color="green", label="Chosen", alpha=0.5, bins=30)
sns.histplot(scores_rejected, color="red", label="Rejected", alpha=0.5, bins=30)
plt.title("Score Distribution: Chosen vs Rejected (after reward training)")
plt.xlabel("Reward Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("score_distribution_results-new.png")
plt.show()