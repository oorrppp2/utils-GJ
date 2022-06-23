import numpy as np

reusables = []
trashes = []

for j in range(50):
    train_data = [i for i in range(300)]
    train_data = np.asarray(train_data) * 0.01
    distributions = np.random.normal(0, 0.1, 300)
    train_data += distributions
    reusables.append(train_data)


for j in range(50):
    train_data = [i for i in range(300)]
    train_data = np.asarray(train_data) ** 2 * 0.01
    distributions = np.random.normal(0, 0.1, 300)
    train_data += distributions
    trashes.append(train_data)


shuffled = np.arange(100)
np.random.shuffle(shuffled)
print(shuffled)

training_data = []
training_label = []
for i in shuffled:
    if i > 49:
        training_data.append(trashes[i-50])
        training_label.append(1)
    else:
        training_data.append(reusables[i])
        training_label.append(0)

print(training_data)
print(training_label)
# print(np.random.shuffle(np.arange(50)))