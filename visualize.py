import json
import matplotlib.pyplot as plt

with open('test_results.json', 'r') as file:
    data = json.load(file)

models = list(data.keys())
scores = [data[model]['scores'] for model in models]
times = [data[model]['times'] for model in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for i, model in enumerate(models):
    ax1.plot(list(range(20)), scores[i], marker='o', label=model)

ax1.set_title('Model Scores')
ax1.set_xlabel('Test Case')
ax1.set_ylabel('Scores')
ax1.set_xticks(list(range(20)))
ax1.legend(loc='upper left')

for i, model in enumerate(models):
    ax2.plot(list(range(20)), times[i], marker='o', label=model)

ax2.set_title('Model Times')
ax2.set_xlabel('Test Case')
ax2.set_ylabel('Time (seconds)')
ax2.set_xticks(list(range(20)))
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
