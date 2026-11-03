import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from .core import Models,QwenAgent,load_image,parse_json,project,save_image,ssim
from .methods import cw,jsma,sta

def clip(x,lo,hi,default):
    try:return max(lo,min(hi,float(x)))
    except (ValueError,TypeError):return default

def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    m=Models(args.models_dir,device,args.blind_checkpoint)
    x=load_image(args.image,args.size).to(device)
    q=QwenAgent(not args.no_llm)
    report={}
    if not args.no_llm:
        report=parse_json(q.ask('Describe image content for robustness testing. JSON only, keys scene, regions, image_properties. Do not infer ground truth authenticity.', 'Analyze this image and its salient regions. Do not claim to know which areas are AI-generated.',x,550))
    baseline,blind_base=m.probability(x,args.target,include_blind=args.evaluate_blind)
    eps=args.epsilon/255.;tau=args.min_ssim
    history=[]; best=None;best_score=-float('inf')
    for iteration in range(args.iterations):
        advice={}
        if not args.no_llm:
            context={'info':report,'baseline_surrogate_probability':baseline,'recent_history':history[-3:],'eps_255':eps*255,'ssim_min':tau}
            advice=parse_json(q.ask('You advise a bounded and authorized local robustness evaluation. Output ONLY JSON with optional numeric keys cw_c, cw_lr, jsma_alpha, jsma_pixels, sta_lr, sta_smooth and mixer_lambda. No prose.',json.dumps(context),max_tokens=350))
        ca=clip(advice.get('cw_c'),.05,10,2.0)
        cl=clip(advice.get('cw_lr'),.001,.05,.01)
        ja=clip(advice.get('jsma_alpha'),.1/255,4/255,1/255)
        jp=int(clip(advice.get('jsma_pixels'),1,32,2))
        sl=clip(advice.get('sta_lr'),.001,.05,.01)
        ss=clip(advice.get('sta_smooth'),0,1,.01)
        lam=clip(advice.get('mixer_lambda'),0,1,.5)
        # Sequential by default: running gradient-bearing models concurrently on one GPU
        # substantially increases VRAM and can make runs nondeterministic.
        candidates={
            'cw':cw(x,m,args.target,eps,args.method_steps,cl,ca),
            'jsma':jsma(x,m,args.target,eps,args.method_steps,ja,jp),
            'sta':sta(x,m,args.target,eps,args.method_steps,sl,ss),
        }
        deltas=[(a-x).detach() for a in candidates.values()]
        # Blind model is EVALUATION-ONLY, never used for optimization: prevents target leakage.
        # Thus lambda is applied to surrogate success vs fidelity, NOT claimed black-box confidence.
        def measure(z):
            prob,_=m.probability(z,args.target)
            similarity=ssim(x,z)
            score=lam*prob+(1-lam)*similarity-args.quality_penalty*max(0,tau-similarity)
            return score,prob,similarity
        trials=[(np.eye(3)[i],name,z) for i,(name,z) in enumerate(candidates.items())]
        weights=np.ones(3)/3
        trials.append((weights,'uniform',project(x,x+sum(float(weights[i])*deltas[i] for i in range(3)),eps)))
        # Randomized simplex hill-climb, strictly bounded to query budget.
        for j in range(args.mixer_trials):
            prop=np.random.dirichlet((1.0+iteration*.1,)*3) if j==0 else np.maximum(0,weights+np.random.normal(0,.2,3))
            if prop.sum()==0:continue
            prop=prop/prop.sum()
            z=project(x,x+sum(float(prop[i])*deltas[i] for i in range(3)),eps)
            trials.append((prop,'blend',z))
            if measure(z)[0]>measure(project(x,x+sum(float(weights[i])*deltas[i] for i in range(3)),eps))[0]:weights=prop
        accepted=[]
        for w,name,z in trials:
            score,pr,sim=measure(z)
            if sim>=tau:
                accepted.append((score,pr,sim,w,name,z))
        if not accepted:
            # Never silently violate SSIM minimum. Original input is always valid.
            accepted=[(*measure(x),[0,0,0],'unchanged',x)]
        chosen=max(accepted,key=lambda item:item[0]);score,pr,sim,w,name,z=chosen
        blind=None
        if args.evaluate_blind:
            _,blind=m.probability(z,args.target,include_blind=True)
        row={'iteration':iteration+1,'method':name,'weights':list(map(float,w)),
             'surrogate_target_probability':pr,'blind_target_probability':blind,
             'ssim':sim,'linf_255':float((z-x).abs().max().item()*255),
             'epsilon_255':eps*255,'tau':tau,'score':score,'advice':advice}
        history.append(row)
        save_image(z,output/f'iteration_{iteration+1:02d}.png')
        print(json.dumps(row))
        if score>best_score:best_score=score;best=z.detach().clone()
        if pr>=args.success_probability:break
        window=history[-args.stagnation_window:]
        if len(window)==args.stagnation_window and max(t['surrogate_target_probability'] for t in window)<args.success_probability and max(t['surrogate_target_probability'] for t in window)-min(t['surrogate_target_probability'] for t in window)<args.stagnation_delta:
            eps=min(args.epsilon_max/255.,eps+args.epsilon_step/255.)
            tau=max(args.ssim_floor,tau-args.ssim_step)
    save_image(best,output/'best.png')
    summary={'model_id':os.getenv('VLLM_MODEL_ID','Qwen/Qwen2.5-VL-32B-Instruct-AWQ'),'device':device,'image':str(args.image),'target_class':args.target,'info':report,'surrogate_baseline':baseline,'blind_baseline':blind_base,'blind_evaluated':args.evaluate_blind,'history':history,'note':'Reimplementation: approximate sparse JSMA; blind detector never participates in optimization; Qwen2.5-VL replaces paper Qwen3 advisors.'}
    (output/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    return summary

def parser():
    p=argparse.ArgumentParser(description='ARMOR research reimplementation, single Qwen VLM')
    p.add_argument('--image',required=True)
    p.add_argument('--models-dir',default='models')
    p.add_argument('--output',default='data/runs/latest')
    p.add_argument('--blind-checkpoint',default=None)
    p.add_argument('--evaluate-blind',action='store_true',help='Only evaluate checkpoint; do not use its outputs to optimize')
    p.add_argument('--target',type=int,choices=[0,1],default=0)
    p.add_argument('--size',type=int,default=224)
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--method-steps',type=int,default=12)
    p.add_argument('--mixer-trials',type=int,default=8)
    p.add_argument('--epsilon',type=float,default=8,help='Linf in 8-bit pixels')
    p.add_argument('--epsilon-max',type=float,default=12)
    p.add_argument('--epsilon-step',type=float,default=1)
    p.add_argument('--min-ssim',type=float,default=.90)
    p.add_argument('--ssim-floor',type=float,default=.80)
    p.add_argument('--ssim-step',type=float,default=.01)
    p.add_argument('--quality-penalty',type=float,default=5)
    p.add_argument('--success-probability',type=float,default=.9)
    p.add_argument('--stagnation-window',type=int,default=2)
    p.add_argument('--stagnation-delta',type=float,default=.005)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-llm',action='store_true',help='Offline test without Qwen')
    p.add_argument('--cpu',action='store_true')
    return p

def main():
    args=parser().parse_args()
    if args.evaluate_blind and not args.blind_checkpoint:raise SystemExit('--evaluate-blind requires --blind-checkpoint')
    if args.size!=224 and args.blind_checkpoint:raise SystemExit('ViT-B/16 needs --size 224')
    if args.epsilon>args.epsilon_max or args.min_ssim<args.ssim_floor:raise SystemExit('Invalid attack budget bounds')
    run(args)
if __name__=='__main__':main()
