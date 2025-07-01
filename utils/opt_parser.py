import torch.optim as optim

from analogAlg.AnalogSGD import AnalogSGD
from analogAlg.Astorm import AStorm
from analogAlg.AstormSaif import AStormS
from analogAlg.Dstorm import DStorm
from analogAlg.HamiltonianDescent import HamiltonianDescent

def get_optimizer(ALG_STR, model, lr):
    if ALG_STR == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif ALG_STR == 'AnalogSGD':
        optimizer = AnalogSGD(model.parameters(), lr=lr, symmetric_point=0, active_radius=tau)
    elif ALG_STR == 'SHD':
        beta = 0.01
        # beta = 0.2
        # beta = 0.1
        optimizer = HamiltonianDescent(model.parameters(), alpha=lr, beta=beta, tau=tau,
                                       update_frequency=1)
    elif ALG_STR == 'STORM':
        beta = 0.2
        optimizer =  AStorm(model.parameters(), lr, beta=beta, c=100, tau=tau)
    elif ALG_STR == 'DSTORM':
        beta = 0.2
        # c = 8e3 # for BS=100 or BS=3
        # c = 8e2 # for BS=3
        c = 1e2
        # c = 10
        optimizer = DStorm(model.parameters(), lr, beta=beta, c=c)
        
        # optimizer = AStormS(model.parameters(),lr,c,tau)
    else:
        assert False, 'Unknown Algorithm'
        
    return optimizer