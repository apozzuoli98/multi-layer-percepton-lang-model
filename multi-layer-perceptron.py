import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import pickle


class MLP:
    def __init__(self):
        # random.seed = (42)
        self.words = open('names.txt', 'r').read().splitlines()
        random.shuffle(self.words)

        self.context = 3 # context length
        
        # build the vocabulary of characters
        self.chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s:i+1 for i,s in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()}

        # build training, dev, and test datasets
        n1 = int(0.8*len(self.words))
        n2 = int(0.9*len(self.words))

        self.Xtr, self.Ytr = self.build_dataset(self.words[:n1], self.context)
        self.Xdev, self.Ydev = self.build_dataset(self.words[n1:n2], self.context)
        self.Xte, self.Yte = self.build_dataset(self.words[n2:], self.context)

        # initialize parameters
        self.parameters = self.init_params()

        # train the mlp
        self.train(self.parameters)

        print('Training loss:')
        self.show_loss(self.parameters, self.Xtr, self.Ytr)

        print('Dev loss:')
        self.show_loss(self.parameters, self.Xdev, self.Ydev)

        print('Test loss:')
        self.show_loss(self.parameters, self.Xte, self.Yte)

        # Save the mlp
        save_data = {
            'parameters': self.parameters,
            'block_size': self.context,
            'itos': self.itos
        }
        with open('mlp.pkl', 'wb') as f:
            pickle.dump(save_data, f)


    def build_dataset(self, words, block_size):
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix] # crop and append

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y
    
    def init_params(self):
        g = torch.Generator()
        C = torch.randn((27, 10), generator=g)
        W1 = torch.randn((30, 300), generator=g) # 30 = 3 * 10
        b1 = torch.randn(300, generator=g)
        W2 = torch.randn((300, 27), generator=g)
        b2 = torch.randn(27, generator=g)
        parameters = [C, W1, b1, W2, b2]
        return parameters
    
    def train(self, parameters):
        for p in parameters:
            p.requires_grad = True

        C = parameters[0]
        W1 = parameters[1]
        b1 = parameters[2]
        W2 = parameters[3]
        b2 = parameters[4]

        # lre = torch.linspace(-3, 0, 1000)
        # lrs = 10**lre

        lri = []
        lossi = []
        stepi = []

        for i in range(200000):

            # minibatch construct
            ix = torch.randint(0, self.Xtr.shape[0], (64,))

            # forward pass
            emb = C[self.Xtr[ix]] # (32, 3, 2)
            h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
            logits = h @ W2 + b2 # (32, 27)
            loss = F.cross_entropy(logits, self.Ytr[ix])

            # backward pass
            for p in parameters:
                p.grad = None
            loss.backward()

            # update
            # lr = lrs[i]
            lr = 0.1 if i < 100000 else 0.01
            for p in parameters:
                p.data += -lr * p.grad

            # track stats
            # lri.append(lre[i])
            # lossi.append(loss.item())
            # stepi.append(i)
            # lossi.append(loss.log10().item())

        # plt.plot(stepi, lossi)
        # plt.show()

    def show_loss(self, parameters, X, Y):
        C = parameters[0]
        W1 = parameters[1]
        b1 = parameters[2]
        W2 = parameters[3]
        b2 = parameters[4]

        emb = C[X]
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
        logits = h @ W2 + b2 # (32, 27)
        loss = F.cross_entropy(logits, Y)
        print(f'{loss:.3f}')



if __name__ == '__main__':
    m = MLP()