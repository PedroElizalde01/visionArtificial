from sklearn import tree
import pandas as pd
from joblib import dump, load

training_dataset = pd.read_csv("hu_moments.csv", header=None)
cols = training_dataset.columns
hu_values_cols = cols[1:]

labels = training_dataset[cols[0]].values.tolist() # This gives me the array of labels
hu_values = training_dataset[hu_values_cols].values.tolist() # This gives me the 2-dim array with Hu moments for each label

print(labels)
print(hu_values)

# entrenamiento
clasificador = tree.DecisionTreeClassifier().fit(hu_values, labels)

# visualización del árbol de decisión resultante
tree.plot_tree(clasificador)

# guarda el modelo en un archivo
dump(clasificador, 'model.joblib')

# en otro programa, se puede cargar el modelo guardado
clasificadorRecuperado = load('model.joblib')