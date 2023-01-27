# Variable-Learning-rate-heuristic-on-AllCNN-network
Learning rate
                                     
    η = 10^-4 + t /T0 * ηmax, if t ≤ T0  

    η = ηmax * cos(π/2 * (t-T0)/(T-T0)) + 10^-6,    if T0 ≤ t ≤ T
    
    T0=T/5
    
    T=50(epochs) x 391(weight updates) = 19550 (for CIFAR-10 with mini-batch of 128)
                                                                
