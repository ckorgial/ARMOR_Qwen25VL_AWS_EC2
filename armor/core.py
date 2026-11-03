import base64, io, json, os, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from skimage.metrics import structural_similarity
from torchvision.models import resnet50, densenet121, vit_b_16

MODEL_ID = os.getenv("VLLM_MODEL_ID", "Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
MEAN = torch.tensor([.485,.456,.406])[None,:,None,None]
STD = torch.tensor([.229,.224,.225])[None,:,None,None]

def load_image(path, size=224):
    im=Image.open(path).convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)[None].float()/255

def image_bytes(x):
    a=(x.detach().cpu().clamp(0,1)[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
    out=io.BytesIO(); Image.fromarray(a).save(out,format='PNG'); return out.getvalue()

def save_image(x,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(image_bytes(x))

def ssim(x,y):
    a=np.asarray(Image.open(io.BytesIO(image_bytes(x))).convert('RGB'))
    b=np.asarray(Image.open(io.BytesIO(image_bytes(y))).convert('RGB'))
    return float(structural_similarity(a,b,channel_axis=2,data_range=255))

def project(x, cand, eps):
    return (x+(cand-x).clamp(-eps,eps)).clamp(0,1).detach()

class Models:
    def __init__(self, directory, device='cpu', target_path=None):
        self.device=torch.device(device)
        self.models={}
        for name in ('resnet50','densenet121'):
            model=self._build(name)
            state=torch.load(str(Path(directory)/f'{name}.pth'),map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True)
            self.models[name]=model.to(self.device).eval().requires_grad_(False)
        self.target=None
        if target_path:
            self.target=self._build('vit_b_16')
            self.target.load_state_dict(torch.load(target_path,map_location='cpu',weights_only=True),strict=True)
            self.target=self.target.to(self.device).eval().requires_grad_(False)
        self.mean=MEAN.to(self.device);self.std=STD.to(self.device)
    @staticmethod
    def _build(name):
        if name=='resnet50':
            m=resnet50(weights=None);m.fc=torch.nn.Linear(m.fc.in_features,2)
        elif name=='densenet121':
            m=densenet121(weights=None);m.classifier=torch.nn.Linear(m.classifier.in_features,2)
        elif name=='vit_b_16':
            m=vit_b_16(weights=None);m.heads.head=torch.nn.Linear(m.heads.head.in_features,2)
        else:raise ValueError(name)
        return m
    def logits(self, x):
        z=(x-self.mean)/self.std
        return [m(z) for m in self.models.values()]
    def loss(self,x,target):
        return torch.stack([F.cross_entropy(z,torch.tensor([target],device=x.device)) for z in self.logits(x)]).mean()
    @torch.no_grad()
    def probability(self,x,target,include_blind=False):
        z=(x.to(self.device)-self.mean)/self.std
        vals=[m(z).softmax(1)[0,target].item() for m in self.models.values()]
        bb=None
        if include_blind and self.target is not None:
            bb=float(self.target(z).softmax(1)[0,target].item())
        return float(np.mean(vals)),bb

class QwenAgent:
    def __init__(self, enabled=True):
        self.enabled=enabled
        if enabled:
            self.client=OpenAI(base_url=os.getenv('VLLM_BASE_URL','http://127.0.0.1:8000/v1'),api_key=os.getenv('VLLM_API_KEY','local'),timeout=180,max_retries=1)
    def ask(self,system,question,image=None,max_tokens=450):
        if not self.enabled:return '{}'
        content=[{'type':'text','text':question}]
        if image is not None:
            url='data:image/png;base64,'+base64.b64encode(image_bytes(image)).decode()
            content.append({'type':'image_url','image_url':{'url':url}})
        response=self.client.chat.completions.create(model=MODEL_ID,messages=[{'role':'system','content':system},{'role':'user','content':content}],temperature=0,max_tokens=max_tokens)
        return response.choices[0].message.content or '{}'

def parse_json(text):
    try:return json.loads(text)
    except (ValueError,TypeError):
        match=re.search(r'\{[\s\S]*\}',text or '')
        if match:
            try:return json.loads(match.group())
            except ValueError:pass
    return {}
