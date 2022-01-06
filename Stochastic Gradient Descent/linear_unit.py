from Perceptron.perceptron import Perceptron

f = lambda x:x
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)

def get_training_dataset():
    input_vecs =[[5], [3], [8], [1.4], [10.1]]
    labels =[5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels