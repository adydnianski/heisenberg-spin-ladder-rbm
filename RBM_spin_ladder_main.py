import numpy as np
import matplotlib.pyplot as plt
import time
import os

# RBM class calulates the wavefunction for given spin configuration sigma
# calculates gradients, saves to file etc

class RBM:
    def __init__(self,n_visible,n_hidden,parameters=None):
        # width of the gaussian from which we initialize random starting parameters
        std_dev = 0.1
        # you can load parameters from a previous run
        if parameters != None:
            self.load_parameters(parameters)
        # the weights are complex
        else:
            self.a = np.random.normal(0, std_dev, n_visible) + 1j * np.random.normal(0, std_dev, n_visible)
            self.b = np.random.normal(0, std_dev, n_hidden) + 1j * np.random.normal(0, std_dev, n_hidden)
            self.W = np.random.normal(0, std_dev, (n_visible, n_hidden)) + 1j * np.random.normal(0, std_dev, (n_visible, n_hidden))

    
    def calc_bWsigma(self,sigma):
        return self.b + np.dot(sigma,self.W)
    
    # for numerical stability calculate the log(psi) instead of psi
    def calc_log_psi(self,sigma):

        p1 = np.dot(self.a, sigma)
        p2 = np.sum(np.log(2 * np.cosh(self.calc_bWsigma(sigma))))

        return p1 + p2
    
    # calculate gradients for given sigma
    def calc_gradient(self,sigma):
        bWsigma = self.calc_bWsigma(sigma)

        grad_a = sigma
        grad_b = np.tanh(bWsigma)
        grad_W = np.outer(sigma,grad_b)

        return (grad_a, grad_b, grad_W)
    
    # save to numpy file
    def save_parameters(self, filename):
        np.savez(filename, a=self.a, b=self.b, W=self.W)
        print(f"Parametry zapisane do {filename}")

    # load from numpy file
    def load_parameters(self, filename):
        data = np.load(filename)
        self.a = data['a']
        self.b = data['b']
        self.W = data['W']
        print(f"Parametry wczytane z {filename}")

# this class handles the lattice - it gives the info which spin is coupled to which
# and how to calc the local energy
# if you want to model a different system, swap out this class
class HeisenbergSpinLadder:
    def __init__(self, L, J, J_perp):
        self.L = L
        self.J = J
        self.J_perp = J_perp
        self.bonds = self._get_bonds()


    # first calculate all the bonds just once
    # it returns a list of tuples, each containing the index of a spin, its neighbor and their coupling
    def _get_bonds(self):
        bonds = []
        L = self.L

        # first calculate for neighbors on the same leg
        for i in range(L):
            next_i = (i+1)%L
            bonds.append((i,next_i,self.J))
            bonds.append((i+L,next_i+L,self.J))

        # here calcuate neighbors on the rungs
        for i in range(L):
            next_i = (i+L)
            bonds.append((i,next_i,self.J_perp))

        
        return bonds
    
    # calculate the local energy according to the hamiltonian
    def compute_local_energy(self,lattice):

        # the diagonal part
        E_diagonal = 0
        for (i,i_next,J) in self.bonds:
            E_diagonal += lattice[i]*lattice[i_next]*J/4

        # the offdiagonal part goes here vvvvvv
        E_offdiagonal = 0
        #for (i,i_next,J) in self.bonds:
        #    pass
        #return E_offdiagonal


        
        return E_diagonal + E_offdiagonal
       
    
# Metropolis class flips spins randomly and checks if it did any good
class MetropolisSampler:

    def __init__(self,rbm):
        self.rbm = rbm

    # you pass spin configuration sigma and it gets processed n_iter times
    def sample(self,sigma):
        # maybe this should be corrected
        n_iter = 1000

        # calc log psi of the starting configuration
        start_log_psi = self.rbm.calc_log_psi(sigma)
        success_count = 0
        # for every iteration
        for n in range(n_iter):

            # choose a spin to flip randomly
            index_spinflip = np.random.randint(0,len(sigma))

            # flip it
            sigma[index_spinflip] *= -1

            # calc log psi of the flipped configuration
            prim_log_psi = self.rbm.calc_log_psi(sigma)
            
            log_ratio = prim_log_psi - start_log_psi
            psi_sq_ratio = np.exp(2* log_ratio.real)

            # checking if better or worse than it was using the ratio of amplitudes of wavefunctions
            # but we employ this randomness which allows us to get out of a local minimum
            # and hopefully land in a global one
            # idk if its a correct approach

            if np.random.rand() < psi_sq_ratio:
                # if successful
                start_log_psi = prim_log_psi
                success_count += 1
            else:
                # if not successful
                # revert the spin flip
                sigma[index_spinflip] *= -1
                
        return sigma,success_count




# main function used to perform machine learning
def machine_learn(learning_rate,ratio_NM,L,J,J_perp,N_epochs,n_samples,load_parameters=None):
    # visible - N, hidden - M
    N = 2*L
    M = N*ratio_NM
    # random starting sigma
    sigma = np.random.choice([-1, 1], size=N)

    # initializing all the classes

    rbm = RBM(N,M,parameters=load_parameters)
    sampler = MetropolisSampler(rbm=rbm)
    ladder = HeisenbergSpinLadder(L,J,J_perp)
    energy_plot = []

    # main loop is iterating over epochs
    # each epoch ends in updating the weights
    for epoch in range(N_epochs):
        # spin configurations
        Sigma_base = []

        # n_samples is the size of the spin configuration basis
        for _ in range(n_samples):
            sigma,success = sampler.sample(sigma)
            Sigma_base.append(sigma.copy())
        
        #calculate the energies and gradients
        energies_epoch = []
        a_grad,b_grad,W_grad = [],[],[]


        for sigma in Sigma_base:
            E_loc = ladder.compute_local_energy(sigma)
            energies_epoch.append(E_loc)

            ga, gb, gW = rbm.calc_gradient(sigma)
            a_grad.append(ga)
            b_grad.append(gb)
            W_grad.append(gW)

        energies_epoch = np.array(energies_epoch)
        a_grad = np.array(a_grad)
        b_grad = np.array(b_grad)
        W_grad = np.array(W_grad)



        # helper function to calculate the partials
        def d_theta(grad):
            energies_reshaped = energies_epoch.reshape([-1] + [1] * (grad.ndim - 1))
            term1 = np.mean(energies_reshaped*grad.conj(),axis=0)
            term2 = np.mean(energies_reshaped)*np.mean(grad.conj(),axis=0)
            return 2*(term1-term2)
        
        # partials
        d_a = d_theta(a_grad)
        d_b = d_theta(b_grad)
        d_W = d_theta(W_grad)

        # updating the weights
        rbm.a -= learning_rate * d_a
        rbm.b -= learning_rate * d_b
        rbm.W -= learning_rate * d_W

        print(f"Epoch {epoch}: Energy = {round(np.mean(energies_epoch),4)}")
        energy_plot.append(np.mean(energies_epoch))
    
    path_graphs = "./graphs/"
    path_params = "./parameter_files/"
    if not os.path.exists(path_graphs):
        os.mkdir(path_graphs)
    if not os.path.exists(path_params):
        os.mkdir(path_params)

    filename = f"RBM_J{J}_Jperp{J_perp}_L{L}_ratioNM{ratio_NM}_learningrate{learning_rate}_Nepochs{N_epochs}_Sigmabase{n_samples}.npz"
    
    rbm.save_parameters(path_params+filename)
    plt.figure(figsize=(8,5))
    plt.scatter(range(N_epochs),energy_plot,color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("<E>")
    plt.title(f"RBM_J{J}_Jperp{J_perp}_L{L}_ratioNM{ratio_NM}_learningrate{learning_rate}_Nepochs{N_epochs}_Sigmabase{n_samples}",fontsize=10)
    plt.savefig(path_graphs+filename[:-4]+".png")
    return path_params+filename


# main
if __name__ == "__main__":



    path_graphs = "./graphs/"


    learning_rate = 0.1

    ratio_NM = 4
    L = 10
    J = 1
    J_perp = 1
    N_epochs = 10
    #n_samples = 1000
    samples = [1000,5000,10000]
    time_list = []
    parameters_filename = None



    for n_samples in samples:
        print(f"Starting RBM... n_samples = {n_samples} \n")
        t0 = time.time()

        # if you set load_parameters to parameters_filename
        # each loop - in this case over the spin configuration size
        # will start from the results of the previous n_samples
        # if its none then it starts from randomness

        parameters_filename = machine_learn(learning_rate,
                                     ratio_NM,
                                     L=L,
                                     J=J,
                                     J_perp=J_perp,
                                     N_epochs=N_epochs,
                                     n_samples=n_samples,
                                     load_parameters=None)
        t1 = time.time()
        dt = round(t1-t0,2)
        print(f"Spin configuration size {n_samples}, time: {dt}s")
        time_list.append(dt)


    plt.figure(figsize=(8,5))
    plt.scatter(samples,time_list,color="tab:blue")
    plt.xlabel("Base size SIGMA")
    plt.ylabel("Learning time")
    plt.title(f"RBM_Nepochs{N_epochs}",fontsize=10)

    g3_filename = f"RBM_Nepochs{N_epochs}_time_vs_Basesize.png"
    plt.savefig(path_graphs+g3_filename)