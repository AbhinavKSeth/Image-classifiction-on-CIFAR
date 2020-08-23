class Relu:
        def __init__(self):
            self.inV = None

        def forward(self,inV):
            outV = np.zeros((len(inV),1))
            for i in range(0,len(outV)):
                outV[i][0] = max(0,inV[i][0])
            self.inV =inV
            return outV

        def backprop(self,grad):
            grad[self.inV<0]= 0;
            return grad
