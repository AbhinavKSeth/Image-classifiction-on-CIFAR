
class CEloss:
        def __init__(self):
            self.label = None
            self.inV = None

        def forward(self,inV,label):
            Loss = -np.log(inV[label][0])
            self.label = label
            self.inV = inV
            return Loss

        def backprop(self,grad):
            grad = np.zeros((self.inV.shape[0],1))
            grad[self.label][0] = -1/self.inV[self.label][0]
            return grad
