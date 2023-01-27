# Variable-Learning-rate-heuristic-on-AllCNN-network

η = 
    10^-4 + t /T0 * ηmax,                                           if t ≤ T0
    ηmax * cos(π/2 * (t-T0)/(T-T0)) + 10^-6,    if T0 ≤ t ≤ T
    10^-6,                                                                       if t > T
