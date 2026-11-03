import torch
import torch.nn.functional as F
from .core import project

def cw(x,models,target,eps,steps=12,lr=.01,c=2.):
    """Projected L2 + targeted CE objective, paper's projected CW variant."""
    d=torch.nn.Parameter(torch.zeros_like(x)); opt=torch.optim.Adam([d],lr=lr)
    for _ in range(steps):
        opt.zero_grad();adv=(x+d).clamp(0,1)
        obj=d.square().mean()+c*models.loss(adv,target)
        obj.backward();opt.step()
        with torch.no_grad():d.copy_(project(x,x+d,eps)-x)
    return project(x,x+d,eps)

def jsma(x,models,target,eps,steps=12,alpha=1/255,pixels=2):
    """Approximate saliency JSMA: targeted gradient ranking, sparse coordinate updates.
    NOT the exact combinatorial pairwise JSMA defined by Eq. (3).
    """
    adv=x.clone()
    for _ in range(steps):
        y=adv.detach().requires_grad_(True)
        z=models.logits(y)
        # Gradient of target vs alternate logits; choose positive saliency locations
        score=torch.stack([(a[0,target]-a[0,1-target]) for a in z]).mean()
        grad=torch.autograd.grad(score,y)[0]
        flat=grad.flatten(); idx=torch.topk(flat.abs(),min(pixels,flat.numel())).indices
        delta=torch.zeros_like(flat);delta[idx]=alpha*flat[idx].sign()
        adv=project(x,adv+delta.view_as(x),eps)
    return adv

def sta(x,models,target,eps,steps=12,lr=.01,smooth=.01,flow_limit=.08):
    """Spatial flow attack using differentiable warp; final Linf projection."""
    _,_,h,w=x.shape
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij')
    base=torch.stack([xx,yy],-1)[None]
    flow=torch.nn.Parameter(torch.zeros((1,h,w,2),device=x.device));opt=torch.optim.Adam([flow],lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        warp=F.grid_sample(x,(base+flow).clamp(-1,1),mode='bilinear',padding_mode='border',align_corners=True)
        reg=(flow[:,1:]-flow[:,:-1]).abs().mean()+(flow[:,:,1:]-flow[:,:,:-1]).abs().mean()
        loss=models.loss(warp,target)+smooth*reg
        loss.backward();opt.step()
        with torch.no_grad():flow.clamp_(-flow_limit,flow_limit)
    return project(x,F.grid_sample(x,(base+flow).clamp(-1,1),padding_mode='border',align_corners=True),eps)
