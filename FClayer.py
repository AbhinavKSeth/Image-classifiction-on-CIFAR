import numpy as np
class FClayer:
        def __init__(self,Sin,Sout):
            self.Sin = Sin
            self.Sout = Sout
            self.W = np.random.randn(Sin,Sout)/np.sqrt(Sin)    # initializing weights and bias matrices with random normal distribition
            self.b = np.random.randn(Sout,1)/np.sqrt(Sin)
            self.inV = None

        def forward(self,inV):
            outV = np.matmul(np.transpose(self.W),inV)+ self.b
            self.inV = inV  #storing the input vector for backpropagation
            return outV

        def backprop(self,grad):
            Lgrad = np.matmul(self.W,grad)  #multiply the input grad  with the gradient wrt input which is the weight matrix
            self.W = self.W - 0.005*np.matmul(self.inV,np.transpose(grad)) #updating the weights and biases with their respective gradients
            self.b = self.b - 0.005*grad
            return Lgrad
