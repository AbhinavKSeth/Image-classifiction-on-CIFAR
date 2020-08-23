class Softmax:
        def __init__(self):
            self.outV = None

        def forward(self,inV):
            outV = np.exp(inV)
            outV = outV/np.sum(outV)
            self.outV = outV
            return outV

        def backprop(self,grad):
            size = self.outV.shape[0]
            Lgrad = np.zeros((size,size))
            for i in range(size):
                for j in range(size):
                    if(i==j):
                        Lgrad[i][i] = self.outV[i][0]*(1-self.outV[i][0])
                    else:
                        Lgrad[i][j] = -self.outV[i][0]*self.outV[j][0]
                        Lgrad[j][i] = -self.outV[i][0]*self.outV[j][0]
            grad = np.matmul(Lgrad,grad)
            return grad
