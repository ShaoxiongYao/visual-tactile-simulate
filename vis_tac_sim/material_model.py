import numpy as np
import torch
from torch.func import jacrev

"""
Material model of deformable objects
Note: this class is intended to be torch-based, input/output should be torch.tensor

FEM parts follow: http://viterbi-web.usc.edu/~jbarbic/femdefo/
"""


class BaseMaterialModel:
    def __init__(self, config=None):
        self.config = config

        self.p_dim = None
        self.u_dim = None
        self.m_dim = None
        self.f_dim = None

        self.geom_type: str = None

    def zero_material(self):
        raise NotImplementedError

    def unit_material(self):
        raise NotImplementedError
    
    def sample_material(self, distribution_parameters):
        raise NotImplementedError

    def element_forces(self, p: torch.Tensor, u: torch.Tensor, material: torch.Tensor) -> torch.Tensor:
        """Compute the forces given the displacement and material parameters."""
        raise NotImplementedError
    
    def element_stiffness(self, p: torch.Tensor, u: torch.Tensor, material: torch.Tensor) -> torch.Tensor:
        """Compute the stiffness matrix given the displacement and material parameters."""
        raise NotImplementedError

    def element_stiffness_autodiff(self, p: torch.Tensor, u: torch.Tensor, material: torch.Tensor) -> torch.Tensor:
        """Compute the stiffness matrix given the displacement and material parameters."""
        return jacrev(self.element_forces, argnums=1)(p, u, material)

class LinearSpringModel(BaseMaterialModel):
    def __init__(self, config=None):
        super().__init__(config)

        self.p_dim = 6
        self.u_dim = 6
        self.m_dim = 1
        self.f_dim = 6
        self.geom_type = "spring"
    
    def zero_material(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def unit_material(self) -> torch.Tensor:
        return torch.tensor(1.0)
    
    def element_forces(self, p, u, material):
        k = material
        u1, u2 = u[:3], u[3:]
        f12 = k*(u1-u2)
        return torch.cat([-f12, f12])

    def element_stiffness(self, p, u, material):
        return self.element_stiffness_autodiff(p, u, material)

class CorotateSpringModel(BaseMaterialModel):
    def __init__(self, config=None):
        super().__init__(config)

        self.p_dim = 6
        self.u_dim = 6
        self.m_dim = 1
        self.f_dim = 6
        self.geom_type = "spring"
    
    def zero_material(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def unit_material(self) -> torch.Tensor:
        return torch.tensor(1.0)

    def element_forces(self, p, u, material):
        k = material
        p1, p2 = p[:3], p[3:]
        u1, u2 = u[:3], u[3:]
        rest_dr = p1-p2
        curr_dr = rest_dr + u1-u2
        rest_l = torch.norm(rest_dr)
        curr_l = torch.norm(curr_dr)
        f12 = k*curr_dr*(1.0 - rest_l/curr_l)
        return torch.cat([-f12, f12])

    def element_stiffness(self, p, u, material):
        return self.element_stiffness_autodiff(p, u, material)

class IsotropicTetraModel(BaseMaterialModel):
    def __init__(self, config=None):
        super().__init__(config)

        self.p_dim = 12
        self.u_dim = 12
        self.m_dim = 2
        self.f_dim = 12
        self.geom_type = "tetra"

    def zero_material(self) -> torch.Tensor:
        return torch.tensor([1.0, 1.0])

    def unit_material(self) -> torch.Tensor:
        return torch.tensor([1.0, 1.0])

    def element_stiffness(self, p, u, material):
        return self.element_stiffness_autodiff(p, u, material)


class LinearTetraModel(IsotropicTetraModel):
    def __init__(self, config=None):
        super().__init__(config)
    
    def element_forces(self, p, u, material):
        mu, lam = material # Lame parameters
        I3 = torch.eye(3)

        p0, p1, p2, p3 = p.split(3, dim=0)
        u0, u1, u2, u3 = u.split(3, dim=0)
        # print('p0, p1, p2, p3:', p0, p1, p2, p3)
        # print('u0, u1, u2, u3:', u0, u1, u2, u3)

        Dm = torch.vstack([p1-p0, p2-p0, p3-p0]).T
        Ds = Dm + torch.vstack([u1-u0, u2-u0, u3-u0]).T
        # print('Dm:', Dm)
        # print('Ds:', Ds)

        Dm_inv = torch.inverse(Dm)

        F = Ds @ Dm_inv
        # print('F:', F)
        P = mu*(F + F.T - 2*I3) + lam*torch.trace(F - I3)*I3

        W = (1/6)*torch.det(Dm).abs()
        # print('W:', W)
        H = -W*P @ Dm_inv.T

        f1, f2, f3 = H.T
        f0 = -(f1+f2+f3)
        return torch.cat([f0, f1, f2, f3])
    
    def element_forces_batch(self, p_batch:torch.Tensor, u_batch:torch.Tensor, m_batch:torch.Tensor):
        mu_batch, lam_batch = m_batch[:, 0], m_batch[:, 1]
        mu_batch = mu_batch.view(-1, 1, 1)
        lam_batch = lam_batch.view(-1, 1, 1)
        I3 = torch.eye(3)

        p0, p1, p2, p3 = p_batch[:, 0, :], p_batch[:, 1, :], p_batch[:, 2, :], p_batch[:, 3, :]
        u0, u1, u2, u3 = u_batch[:, 0, :], u_batch[:, 1, :], u_batch[:, 2, :], u_batch[:, 3, :]

        # print('p0, p1, p2, p3:', p0, p1, p2, p3)
        # print('u0, u1, u2, u3:', u0, u1, u2, u3)
    
        Dm = torch.stack([p1-p0, p2-p0, p3-p0], dim=1)
        Ds = Dm + torch.stack([u1-u0, u2-u0, u3-u0], dim=1)
        Dm = Dm.transpose(1, 2)
        Ds = Ds.transpose(1, 2)
        # print('Dm:', Dm)
        # print('Ds:', Ds)

        Dm_inv = torch.inverse(Dm)

        F = Ds @ Dm_inv
        # print('F:', F)

        P = mu_batch*(F + F.transpose(1, 2) - 2*I3) + \
            lam_batch*torch.einsum('bii->b', F - I3).view(-1, 1, 1)*I3[None, :, :]
        # print('P:', P)

        W = (1/6)*torch.det(Dm).abs()
        W = W.view(-1, 1, 1)
        # print('W:', W)
        H = -W*P @ Dm_inv.transpose(1, 2)

        HT = H.transpose(1, 2)
        f1, f2, f3 = HT[:, 0, :], HT[:, 1, :], HT[:, 2, :]
        f0 = -(f1+f2+f3)

        # print('f0:', f0.shape)
        # print('f1:', f1.shape)
        # print('f2:', f2.shape)
        # print('f3:', f3.shape)

        return torch.cat([f0, f1, f2, f3], dim=1)

class CorotateTetraModel(IsotropicTetraModel):
    def __init__(self, config=None):
        super().__init__(config)

    def element_forces(self, p, u, material):
        mu, lam = material # Lame parameters
        I3 = torch.eye(3)

        p0, p1, p2, p3 = p.split(3, dim=0)
        u0, u1, u2, u3 = u.split(3, dim=0)

        Dm = torch.vstack([p1-p0, p2-p0, p3-p0]).T
        Ds = Dm + torch.vstack([u1-u0, u2-u0, u3-u0]).T

        Dm_inv = torch.inverse(Dm)

        F = Ds @ Dm_inv

        U, _, Vh = torch.linalg.svd(F)
        R = (U @ Vh).detach() # NOTE: disable grad w.r.t. svd

        P = mu*(F + R @ F.T @ R - R) + lam*torch.trace(R.T@F - I3)*R

        W = (1/6)*torch.det(Dm).abs()
        H = -W*P @ Dm_inv.T

        f1, f2, f3 = H.T
        f0 = -(f1+f2+f3)
        return torch.cat([f0, f1, f2, f3])


class StVKTetraModel(IsotropicTetraModel):
    def __init__(self, config=None):
        super().__init__(config)
    
    def element_forces(self, p, u, material):
        mu, lam = material # Lame parameters
        I3 = torch.eye(3)

        p0, p1, p2, p3 = p.split(3, dim=0)
        u0, u1, u2, u3 = u.split(3, dim=0)

        Dm = torch.vstack([p1-p0, p2-p0, p3-p0]).T
        Ds = Dm + torch.vstack([u1-u0, u2-u0, u3-u0]).T

        Dm_inv = torch.inverse(Dm)

        F = Ds @ Dm_inv
        
        E = 0.5*(F.T @ F - I3)
        P = F @ (2*mu*E + lam*torch.trace(E)*I3)

        W = (1/6)*torch.det(Dm).abs()
        H = -W*P @ Dm_inv.T

        f1, f2, f3 = H.T
        f0 = -(f1+f2+f3)
        return torch.cat([f0, f1, f2, f3])
