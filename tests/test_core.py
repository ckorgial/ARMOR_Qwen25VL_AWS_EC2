import torch
from armor.core import project,ssim
from armor.runner import clip

def test_projection():
    x=torch.full((1,3,8,8),.5)
    z=project(x,x+1,8/255)
    assert (z-x).abs().max().item() <= 8/255+1e-6

def test_identical_ssim():
    x=torch.rand(1,3,32,32)
    assert ssim(x,x)==1.0

def test_clip():
    assert clip('999',0,1,.5)==1
