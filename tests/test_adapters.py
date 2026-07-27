from pathlib import Path
import yaml
from ltx.adapters.cm import CMAdapter
from ltx.adapters.oc import OCAdapter
from ltx.config import Task


def task(tmp_path, adapter):
    return Task(id='x',campaign='c',stage='s',adapter=adapter,method=adapter,seed=2,priority=1,
      dataset={'root':str(tmp_path/'data'),'data_type':'cifar100lt' if adapter=='cm' else 'cifar10lt','num_class':10,'imbalance_factor':.01},
      train={'total_steps':300001 if adapter=='cm' else 200000,'batch_size':64 if adapter=='cm' else 128},
      eval={'checkpoint_step':300000 if adapter=='cm' else 200000,'num_images':50000},method_config={},repository={},
      runtime={'repos_root':str(tmp_path/'repos'),'python':'python','data_root':str(tmp_path/'data')},retry={},run_dir=str(tmp_path/'run'))


def test_cm_uses_only_documented_cli(tmp_path):
    repo=tmp_path/'repos/cm/configs/cifar100lt_ir100'; repo.mkdir(parents=True)
    cfg={'seed':0,'output_dir':'x','dataset':{'root':'./data'},'training':{'batch_size':64,'total_steps':300001},'evaluation':{}}
    (repo/'cm.yaml').write_text(yaml.safe_dump(cfg))
    phases=CMAdapter(Path(__file__).resolve().parents[1]).phases(task(tmp_path,'cm'))
    assert phases[0].command[-2:]==['--config',str(tmp_path/'run/cm.resolved.yaml')]
    assert '--output_dir' not in phases[0].command
    assert [p.name for p in phases]==['train','sample','metrics_cm_cifar_lt']


def test_cm_uses_native_checkpoint_step_resume(tmp_path):
    t = task(tmp_path, 'cm'); run = Path(t.run_dir); run.mkdir(parents=True)
    (run/'ckpt_42.pt').touch()
    repo = tmp_path/'repos/cm/configs/cifar100lt_ir100'; repo.mkdir(parents=True)
    cfg={'seed':0,'output_dir':'x','dataset':{'root':'./data'},'training':{'batch_size':64,'total_steps':300001},'evaluation':{}}
    (repo/'cm.yaml').write_text(yaml.safe_dump(cfg))
    phases = CMAdapter(Path(__file__).resolve().parents[1]).phases(t)
    assert phases[0].command[-2:] == ['--ckpt_step', '42']
    assert '--resume_ckpt' not in phases[0].command


def test_cm_can_resume_the_step_zero_checkpoint(tmp_path):
    t = task(tmp_path, 'cm'); run = Path(t.run_dir); run.mkdir(parents=True)
    (run/'ckpt_0.pt').touch()
    repo = tmp_path/'repos/cm/configs/cifar100lt_ir100'; repo.mkdir(parents=True)
    cfg={'seed':0,'output_dir':'x','dataset':{'root':'./data'},'training':{'batch_size':64,'total_steps':300001},'evaluation':{}}
    (repo/'cm.yaml').write_text(yaml.safe_dump(cfg))
    phases = CMAdapter(Path(__file__).resolve().parents[1]).phases(t)
    assert phases[0].command[-2:] == ['--ckpt_step', '0']


def test_oc_explicit_steps_and_resume(tmp_path):
    t=task(tmp_path,'oc'); run=Path(t.run_dir); run.mkdir(parents=True); (run/'ckpt_100000.pt').touch()
    phases=OCAdapter(Path(__file__).resolve().parents[1]).phases(t)
    cmd=phases[0].command
    assert '--total_steps=200001' in cmd and '--resume' in cmd and '--ckpt_step=100000' in cmd
