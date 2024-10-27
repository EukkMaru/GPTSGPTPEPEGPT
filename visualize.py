import json
import matplotlib.pyplot as plt

with open('test_results.json', 'r') as file:
    data = json.load(file)

models = list(data.keys())
scores = [data[model]['scores'] for model in models]
times = [data[model]['times'] for model in models]

fig, ((ax1_1, ax2_1), (ax1_2, ax2_2)) = plt.subplots(2, 2, figsize=(12, 12))

for i, model in enumerate(models):
    if i == 0:
        ax1_1.plot(list(range(20)), scores[i], marker='o', label=model)
        ax1_2.plot(list(range(20)), scores[i], marker='o', label=model)
        continue
    ax1_1.plot(list(range(20)), scores[i], marker='o', label=model) if i in [1, 2] else ax1_2.plot(list(range(20)), scores[i], marker='o', label=model)

ax1_1.set_title('Model Scores')
ax1_1.set_xlabel('Test Case')
ax1_1.set_ylabel('Scores')
ax1_1.set_xticks(list(range(20)))
ax1_1.legend(loc='upper left')

ax1_2.set_xlabel('Test Case')
ax1_2.set_ylabel('Scores')
ax1_2.set_xticks(list(range(20)))
ax1_2.legend(loc='upper left')

for i, model in enumerate(models):
    if i == 0:
        ax2_1.plot(list(range(20)), times[i], marker='o', label=model)
        ax2_2.plot(list(range(20)), times[i], marker='o', label=model)
        continue
    ax2_1.plot(list(range(20)), times[i], marker='o', label=model) if i in [1, 2] else ax2_2.plot(list(range(20)), times[i], marker='o', label=model)

ax2_1.set_title('Model Times')
ax2_1.set_xlabel('Test Case')
ax2_1.set_ylabel('Time (seconds)')
ax2_1.set_xticks(list(range(20)))
ax2_1.legend(loc='upper left')

ax2_2.set_xlabel('Test Case')
ax2_2.set_ylabel('Time (seconds)')
ax2_2.set_xticks(list(range(20)))
ax2_2.legend(loc='upper left')

plt.tight_layout()
plt.show()
