import json
from pathlib import Path
import numpy as np
from ltx.config import LoadedCampaign, Task
from ltx.eval.aggregate import aggregate


def make_campaign(tmp_path, kill=False):
    src=Path(__file__).resolve().parents[1]
    (tmp_path/'contracts').mkdir(); (tmp_path/'contracts/semantic_metrics.schema.json').write_text((src/'contracts/semantic_metrics.schema.json').read_text())
    arms={'lt':(.030,.30),'oracle':(.015,.42),'predictive':(.020,.39),'permutation':(.029,.31),'pointfit':(.023,.37)}
    if kill: arms['predictive']=(.031,.29)
    tasks=[]; rng=np.random.default_rng(4)
    for arm,(js,rare) in arms.items():
      for seed in range(3):
        run=tmp_path/'runs'/'c'/'decisive_semantic_gate'/arm/f'seed_{seed}'; run.mkdir(parents=True)
        b=1000; draws=run/'draws.npz'
        np.savez(draws,js=rng.normal(js,.0005,b),rare_mode_mass=rng.normal(rare,.003,b),
                 coarse_consistency=rng.normal(.80,.002,b),memorization=rng.normal(.01,.0002,b),
                 FID=rng.normal(8.0 if arm!='predictive' else 8.1,.05,b),KID=rng.normal(.002,.00005,b),Recall=rng.normal(.55,.003,b))
        payload={'js':js,'rare_mode_mass':rare,'coarse_consistency':.80,'memorization':.01,'num_generated':50000,
                 'evaluator_fingerprint':'x','bootstrap_draws_file':'draws.npz','generation':{'FID':8.0 if arm!='predictive' else 8.1,'KID':.002,'Recall':.55}}
        (run/'semantic_metrics.json').write_text(json.dumps(payload))
        tasks.append(Task(id=f'{arm}{seed}',campaign='c',stage='decisive_semantic_gate',adapter='coral',method=arm,seed=seed,priority=1,
          dataset={},train={},eval={},method_config={},repository={},runtime={},retry={},run_dir=str(run)))
    raw={'campaign':{'name':'c'},'aggregation':{'semantic_primary_stage':'decisive_semantic_gate','baseline_arm':'lt','oracle_arm':'oracle',
       'inferred_arm':'predictive','permutation_arm':'permutation','pointfit_arm':'pointfit','bootstrap_repetitions':2000,'confidence_level':.95,
       'noninferiority':{'coarse_consistency_absolute_drop':.01,'memorization_absolute_increase':.01,'fid_relative_increase':.05,'kid_relative_increase':.1,'recall_absolute_drop':.01}}}
    server={'runtime':{'runs_root':str(tmp_path/'runs'),'wandb_mode':'disabled'}}
    return LoadedCampaign(tmp_path,tmp_path/'x.yaml',raw,server,tasks)


def test_aggregate_pass(tmp_path): assert aggregate(make_campaign(tmp_path))['verdict']['status']=='PASS'
def test_aggregate_kill(tmp_path): assert aggregate(make_campaign(tmp_path,True))['verdict']['status']=='KILL'
