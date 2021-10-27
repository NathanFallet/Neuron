# Classe du Neuron
class Neuron:

    # Constructeur
    def __init__(self, weights) -> None:
        self.rate = 0.01
        self.w = weights

    # Lance une prédiction
    # Prends une liste d'entrée et créé la sortie
    def predict(self, inputs):
        # Check la taille de l'entrée
        if len(inputs) != len(self.w) - 1:
            raise Exception("Taille de l'entrée invalide !")

        # Calcul du product scalaire
        return sum(
            map(
                lambda x : self.w[x[0] + 1] * x[1],
                enumerate(inputs)
            )
        ) + self.w[0]

    # Entrairement avec un set de données
    # Prend une liste qui contient des couples liste d'entrée + valeur attendue
    # ainsi que le nombre de répétition
    def train(self, data, repetitions):
        # On effectue les répétitions
        for _ in range(repetitions):
            for row in data:
                # Extraction des données de la ligne
                inputs, expected = row

                # Vérification de la taille
                if len(inputs) == len(self.w) - 1:
                    # On calcul la prédiction
                    prediction = self.predict(inputs)

                    # On récupère l'erreur et on ajuste
                    error = expected - prediction
                    self.w[0] = self.w[0] + self.rate * error
                    for k in range(len(inputs)):
                        self.w[k+1] = self.w[k+1] + self.rate * error * inputs[k]

# Les tests
# On va lui apprendre à faire 2x + 5

data = [([x], 2 * x + 5) for x in range(10)]

neuron = Neuron([1.0, 1.0])
neuron.train(data, 500)

print(neuron.predict([20])) # 45