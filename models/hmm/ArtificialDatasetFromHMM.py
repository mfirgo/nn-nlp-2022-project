from models.hmm.HiddenMarkovModel import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ArtificialDataset(torch.utils.data.Dataset):
    def __init__(self, distributions, device, T=1000, sequence_length=100,transition_matrix=None):
        self.device = device
        self.sequence_length = sequence_length
        self.T = T
        self.MyHMM = HMM(len(distributions), distributions,transition_matrix=transition_matrix)
        self.data_dim = distributions[0].sample().shape[0]
        self.generate_sequence()

    def generate_sequence(self): 
        self.train_X = torch.zeros((self.T, self.sequence_length, self.data_dim))
        self.train_Z = torch.zeros((self.T, self.sequence_length))
        for i in range(self.T):
            x, z = self.MyHMM.sample(self.sequence_length)
            self.train_X[i, :, :] = x
            self.train_Z[i, :] = torch.tensor(z, dtype=torch.int)

        self.test_T = int(0.2*self.T)
        self.test_X = torch.zeros((self.test_T, self.sequence_length, self.data_dim))
        self.test_Z = torch.zeros((self.test_T, self.sequence_length))
        for i in range(self.test_T):
            x, z = self.MyHMM.sample(self.sequence_length)
            self.test_X[i, :, :] = x
            self.test_Z[i, :] = torch.tensor(z, dtype=torch.int)

    def __len__(self):
        return self.T

    def __getitem__(self, index):
        t = self.train_X[index]
        t.to(device)
        return t