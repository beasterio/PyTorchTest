import torch

class Container(torch.nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self, "geometry_parameters.pth", torch.load('..\\input\\geometry_parameters.pth', map_location=torch.device('cpu') ))
        setattr(self, "coords.pth", torch.load('..\\input\\coords.pth', map_location=torch.device('cpu') ))
        setattr(self, "bps.pth", torch.load('..\\input\\bps.pth', map_location=torch.device('cpu') ))
        setattr(self, "spheres.pth", torch.load('..\\input\\spheres.pth', map_location=torch.device('cpu') ))
        setattr(self, "sd_ground_truth.pth", torch.load('..\\input\\sd_ground_truth.pth', map_location=torch.device('cpu') ))
        setattr(self, "grad_gt.pth", torch.load('..\\input\\grad_gt.pth', map_location=torch.device('cpu') ))

container = torch.jit.script(Container())
container.save("packed_data.pt")