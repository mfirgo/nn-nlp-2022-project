import torch
import numpy as np

class HMM(torch.nn.Module):
#   """
#   Hidden Markov Model with discrete observations.
#   """
    def __init__(self, N, distributions, transition_matrix=None, state_priors='uniform'):
        super(HMM, self).__init__()
        #self.M = M # number of possible observations
        self.N = N # number of states

        # A
        self.transition_model = TransitionModel(self.N, transition_matrix)

        # b(x_t)
        self.emission_model = EmissionModel(self.N, distributions)

        # pi # CHECK
        if state_priors=="uniform":
            self.unnormalized_state_priors = torch.ones(self.N)/self.N#torch.nn.Parameter(torch.randn(self.N))#torch.randn(self.N)#
            self.normalized_state_priors = self.unnormalized_state_priors
            self.log_normalized_state_priors = torch.log(self.unnormalized_state_priors)
        elif state_priors=="random":
            self.unnormalized_state_priors = torch.randn(self.N)
            self.normalized_state_priors = torch.nn.functional.softmax(self.unnormalized_state_priors, dim=0)
            self.log_normalized_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
        elif torch.is_tensor(state_priors):
            self.unnormalized_state_priors = state_priors
            self.normalized_state_priors = torch.nn.functional.normalize(self.unnormalized_transition_matrix, p=1, dim=0)
            self.log_normalized_state_priors = torch.log(self.normalized_state_priors)
        else:
            raise ValueError("state_priors must be 'uniform', 'random' or torch tensor")


        # use the GPU
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda: self.cuda()

class TransitionModel(torch.nn.Module):
    def __init__(self, N, transition_matrix=None):
        super(TransitionModel, self).__init__()
        self.N = N
        if transition_matrix is None:
            self.unnormalized_transition_matrix = torch.nn.functional.softmax(torch.randn(N,N), dim=1)#torch.nn.Parameter(torch.randn(N,N))# CHECK
        else:
            self.unnormalized_transition_matrix = transition_matrix
            
    def normalized_transition_matrix(self):
        #return torch.nn.functional.softmax(self.unnormalized_transition_matrix, dim=1) ## CHECK # original dim=0
        return torch.nn.functional.normalize(self.unnormalized_transition_matrix, p=1, dim=1)
    def log_normalized_transition_matrix(self):
        #return torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=1) ## CHECK 
        return torch.log(self.normalized_transition_matrix())

class EmissionModel(torch.nn.Module):
    def __init__(self, N, distributions):
        super(EmissionModel, self).__init__()
        self.N = N
        self.distributions = distributions ## list of distributions

    def pdf(self, hidden_state, observation):
        current_distribution = self.distributions[hidden_state]
        return torch.exp(current_distribution.log_prob(torch.Tensor(observation)))

def sample(self, T=10):
    state_priors = self.normalized_state_priors#torch.nn.functional.softmax(self.unnormalized_state_priors, dim=0)
    transition_matrix = self.transition_model.normalized_transition_matrix()
    #emission_matrix = torch.nn.functional.softmax(self.emission_model.unnormalized_emission_matrix, dim=1)

    # sample initial state
    z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
    z = []; x = []
    z.append(z_t)
    for t in range(0,T):
        # sample emission
        # x_t = torch.distributions.categorical.Categorical(emission_matrix[z_t]).sample().item()
        current_distribution = self.emission_model.distributions[z_t]
        x_t = current_distribution.sample()
        x.append(x_t)

        # sample transition
        z_t = torch.distributions.categorical.Categorical(transition_matrix[z_t, :]).sample().item() # CHECK # original [:, z_t]
        if t < T-1: z.append(z_t)
 
    return torch.stack(x), z

# Add the sampling method to our HMM class
HMM.sample = sample

def HMM_forward(self, x, T, save_log_alpha=True):
    """
    x : IntTensor of shape (batch size, T_max)
    T : IntTensor of shape (batch size)

    Compute log p(x) for each example in the batch.
    T = length of each example
    """
    if self.is_cuda:
        x = x.cuda()
        T = T.cuda()

    batch_size = x.shape[0]; T_max = x.shape[1]
    #log_state_priors = torch.log(self.unnormalized_state_priors)  # TODO #torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
    #log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
    log_state_priors = self.log_normalized_state_priors
    log_alpha = torch.zeros(batch_size, T_max, self.N) # table (sample, t, state) containing log probability of observations from sample to time t and being in state (in time t)
    if self.is_cuda: log_alpha = log_alpha.cuda()

    log_alpha[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors # emission_model - log prob for each distr
    for t in range(1, T_max):
    #print(f"t={t} ", self.emission_model(x[:,t]), self.transition_model(log_alpha[:, t-1, :]))
        log_alpha[:, t, :] = self.emission_model(x[:,t]) + self.transition_model(log_alpha[:, t-1, :])

    if save_log_alpha:
        self.log_alpha = log_alpha
        self.x = x
    # Select the sum for the final timestep (each x may have different length).
    #print("alpha\n", log_alpha)
    log_sums = log_alpha.logsumexp(dim=2)
    #print("log_sums\n", log_sums)
    #log_probs = torch.gather(log_sums, 1, T.view(1,-1))
    log_probs = torch.gather(log_sums, 1, T.view(-1,1)-1)
    return log_probs

def emission_model_forward(self, x_t): ## TODO
    #out = self.distributions.log_prob(x_t)
    #out = 
    out  = []
    for state in range(self.N):
        out.append( self.distributions[state].log_prob(x_t) )
    result = torch.stack(out, dim = 1)
    #print("emission probs\n",result)
    return result

def transition_model_forward(self, log_alpha):
    """
    log_alpha : Tensor of shape (batch size, N)
    Multiply previous timestep's alphas by transition matrix (in log domain)
    """
    log_transition_matrix = self.log_normalized_transition_matrix()

    # Matrix multiplication in the log domain
    out = log_domain_matmul(log_transition_matrix.transpose(0,1), log_alpha.transpose(0,1)).transpose(0,1) # CHECK # original log_transition_matrix
    return out

def log_domain_matmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]#; print(log_A.shape, log_B.shape)
    n = log_A.shape[1]
    p = log_B.shape[1]
    #print(log_A.shape, log_B.shape)
    # log_A_expanded = torch.stack([log_A] * p, dim=2)
    # log_B_expanded = torch.stack([log_B] * m, dim=0)
    # fix for PyTorch > 1.5 by egaznep on Github:
    log_A_expanded = torch.reshape(log_A, (m,n,1))#; print(log_A_expanded.shape)
    log_B_expanded = torch.reshape(log_B, (1,n,p))#; print(log_B_expanded.shape)

    elementwise_sum = log_A_expanded + log_B_expanded #; print("hello")
    out = torch.logsumexp(elementwise_sum, dim=1)#;print(out.shape)

    return out

TransitionModel.forward = transition_model_forward
EmissionModel.forward = emission_model_forward
HMM.forward = HMM_forward

def viterbi(self, x, T):
    """
    x : IntTensor of shape (batch size, T_max)
    T : IntTensor of shape (batch size)
    Find argmax_z log p(x|z) for each (x) in the batch.
    """
    if self.is_cuda:
        x = x.cuda()
        T = T.cuda()

    batch_size = x.shape[0]; T_max = x.shape[1]
    log_state_priors = self.log_normalized_state_priors#torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
    log_delta = torch.zeros(batch_size, T_max, self.N).float()
    psi = torch.zeros(batch_size, T_max, self.N).long()
    if self.is_cuda:
        log_delta = log_delta.cuda()
        psi = psi.cuda()

    log_delta[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors
    for t in range(1, T_max):
        max_val, argmax_val = self.transition_model.maxmul(log_delta[:, t-1, :])
        log_delta[:, t, :] = self.emission_model(x[:,t]) + max_val
        psi[:, t, :] = argmax_val

    # Get the log probability of the best path
    log_max = log_delta.max(dim=2)[0]
    best_path_scores = torch.gather(log_max, 1, T.view(-1,1) - 1)

    # This next part is a bit tricky to parallelize across the batch,
    # so we will do it separately for each example.
    z_star = []
    for i in range(0, batch_size):
        z_star_i = [ log_delta[i, T[i] - 1, :].max(dim=0)[1].item() ]
        for t in range(T[i] - 1, 0, -1):
            z_t = psi[i, t, z_star_i[0]].item()
            z_star_i.insert(0, z_t)

        z_star.append(z_star_i)

    return z_star, best_path_scores # return both the best path and its log probability

def transition_model_maxmul(self, log_alpha):
    log_transition_matrix = self.log_normalized_transition_matrix()#torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

    out1, out2 = maxmul(log_transition_matrix.transpose(0,1), log_alpha.transpose(0,1)) # CHECK # original log_transition_matrix
    return out1.transpose(0,1), out2.transpose(0,1)

def maxmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix

    Similar to the log domain matrix multiplication,
    this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = torch.stack([log_A] * p, dim=2)
    log_B_expanded = torch.stack([log_B] * m, dim=0)

    elementwise_sum = log_A_expanded + log_B_expanded
    out1,out2 = torch.max(elementwise_sum, dim=1)

    return out1,out2

TransitionModel.maxmul = transition_model_maxmul
HMM.viterbi = viterbi

def HMM_backward(self, x, T, save_log_beta=True):
    """
    x : IntTensor of shape (batch size, T_max)
    T : IntTensor of shape (batch size)

    Compute backward log p(x) for each example in the batch.
    T = length of each example
    """
    if self.is_cuda:
        x = x.cuda()
        T = T.cuda()

    batch_size = x.shape[0]; T_max = x.shape[1] - 1
    gather_indexes = torch.zeros((batch_size,1), dtype=torch.int64)
    if self.is_cuda:
        gather_indexes = gather_indexes.cuda()
    #log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
    log_beta = torch.zeros(batch_size, T_max+1, self.N) # table (sample, t, state) containing log probability of observations from sample from time t+1 to T_max and being in state (in time t)
    if self.is_cuda: log_beta = log_beta.cuda()

    log_transition_matrix = self.transition_model.log_normalized_transition_matrix() #torch.nn.functional.log_softmax(self.transition_model.unnormalized_transition_matrix, dim=0)

    log_beta[:, T_max, :] = 0 #1 #self.emission_model(x[:,0]) + log_state_priors # emission_model - log prob for each distr
    for t in range(T_max-1, 0-1, -1):
        suma = (self.emission_model(x[:,t+1])+log_beta[:, t+1, :]).transpose(1,0)
        #print(suma.shape)
        #suma = suma.unsqueeze(1)
        #print(suma.shape)
        out = log_domain_matmul(log_transition_matrix, suma).transpose(1,0)# CHECK # original log_transition_matrix
        #print(out.shape, log_beta[:, t, :].shape)
        log_beta[:, t, :] = out

    if save_log_beta:
        self.log_beta = log_beta
        self.x = x

    log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
    termination = self.emission_model(x[:,0]) + log_state_priors + log_beta[:, 0, :]

    log_sums = log_beta.logsumexp(dim=2)
    #print("log_sums\n", log_sums)
    #log_probs = torch.gather(log_sums, 1, T.view(1,-1))
    #log_probs = torch.gather(log_sums, 1, gather_indexes)
    log_probs = termination.logsumexp(dim=1)
    return log_probs

# def transition_model_backward(self, log_beta):
#   """
#   log_alpha : Tensor of shape (batch size, N)
#   Multiply previous timestep's alphas by transition matrix (in log domain)
#   """
#   log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

#   # Matrix multiplication in the log domain
#   #out = log_domain_matmul(log_transition_matrix, log_alpha.transpose(0,1)).transpose(0,1)
#   out = log_domain_matmul(log_transition_matrix.transpose(0,1), log_beta.transpose(0,1)).transpose(0,1)
#   return out

#TransitionModel.back = transition_model_backward
HMM.back = HMM_backward

def forward_backward_step(self, x, T):
    T_max=10
    forward_result = self.forward(x, T, save_log_alpha=True)
    backward_result = self.back(x, T, save_log_beta=True)
    #log_beta and alpha have shape: (batch_size, T_max, self.N)
    denominator_sum = (self.log_alpha+self.log_beta).logsumexp(dim=2)[:, 0:T_max-1]
    log_alpha = self.log_alpha[:, 0:(T_max-1),:].unsqueeze(3)
    log_beta = self.log_beta[:, 1:T_max, :].unsqueeze(2)
    dim1,dim2,dim3 = x.shape
    log_b = torch.zeros(((dim1,dim3,dim2)))
    for i in range(dim1):
        log_b[i, :] = self.emission_model(x[i,:]).transpose(0,1)
    log_b = log_b.transpose(1,2)[:, 1:T_max, :].unsqueeze(2)

    log_transition_matrix = self.transition_model.log_normalized_transition_matrix()#torch.nn.functional.log_softmax(self.transition_model.unnormalized_transition_matrix, dim=0).unsqueeze(0).unsqueeze(1)
    #print(log_transition_matrix.shape, log_b.shape, log_transition_matrix.shape, log_alpha.shape)

    nominator = log_alpha+log_transition_matrix+log_beta+log_b # CHECK
    #print(nominator.shape)
    log_ksi = nominator - denominator_sum[:, 0:T_max-1].unsqueeze(2).unsqueeze(3)
    

    approx_log_A = log_ksi.logsumexp(dim=(0,1)) 
    approx_log_A = approx_log_A - log_ksi.logsumexp(dim=(0,1,3)).unsqueeze(1)
    approx_A = torch.exp(approx_log_A)#.transpose(0,1)#### Czy chcemy ten transpose CHECK originaly no transpose
    self.transition_model.unnormalized_transition_matrix = approx_A

    return approx_A


HMM.forward_backward_step = forward_backward_step