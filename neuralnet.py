class neuralnet:
        def __init__(self):
            self.Layers = np.array([])

        def forward(self,inV,label):
            outV = inV
            for i in range(0,len(self.Layers)-1):
                outV = self.Layers[i].forward(outV)
            outV = self.Layers[len(self.Layers)-1].forward(outV,label)
            return outV

        def backprop(self):
            grad = np.array([])
            for i in range(len(self.Layers)-1,-1,-1):
                grad = self.Layers[i].backprop(grad)
               # print(grad.shape)
            return

        def addlayer(self,layer):
            self.Layers = np.append(self.Layers,layer)
