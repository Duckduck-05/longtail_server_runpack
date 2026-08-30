import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from ltx.adapters.cm import CMAdapter
from ltx.adapters.coral import CoralAdapter
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


def test_coral_eval_uses_the_cbdm_balanced_fid_cache(tmp_path, monkeypatch):
    t = task(tmp_path, 'coral')
    t.dataset['data_type'] = 'cifar100lt'
    monkeypatch.delenv('LTX_METRICS_ROOT', raising=False)

    phases = CoralAdapter(Path(__file__).resolve().parents[1]).phases(t)
    eval_phase = next(phase for phase in phases if phase.name == 'eval_w1.0')
    eval_cmd = eval_phase.command
    expected_root = (tmp_path / 'repos' / 'CBDM-pytorch' / 'stats').resolve()

    assert f"--fid_cache={expected_root / 'cifar100.train.npz'}" in eval_cmd
    assert eval_phase.env['LTX_METRICS_ROOT'] == str(expected_root)


def test_coral_uses_a_configured_metrics_root_for_fid_and_paper_metrics(tmp_path, monkeypatch):
    t = task(tmp_path, 'coral')
    t.dataset['data_type'] = 'cifar100lt'
    t.eval['paper_metrics'] = True
    configured = tmp_path / 'custom-metrics'
    monkeypatch.setenv('LTX_METRICS_ROOT', str(configured))

    phases = CoralAdapter(Path(__file__).resolve().parents[1]).phases(t)
    eval_phase = next(phase for phase in phases if phase.name == 'eval_w1.0')
    metrics_phase = next(phase for phase in phases if phase.name == 'paper_metrics_w1.0')

    assert f'--fid_cache={configured / "cifar100.train.npz"}' in eval_phase.command
    assert str(configured) in metrics_phase.command
    assert metrics_phase.env['LTX_METRICS_ROOT'] == str(configured)


def test_coral_main_requires_an_explicit_matching_fid_cache():
    source = (Path(__file__).resolve().parents[1] / 'third_party/coral-lt-diffusion/main.py').read_text(
        encoding='utf-8')

    assert "flags.DEFINE_string('fid_cache', ''," in source
    assert 'def resolve_fid_cache():' in source
    assert 'if not FLAGS.fid_cache:' in source
    assert 'if not fid_cache.is_absolute():' in source
    assert 'if not fid_cache.is_file():' in source
    assert 'fid_cache.name != expected_name' in source
    assert "FLAGS.fid_cache = './stats/" not in source


def test_coral_main_rejects_relative_or_wrong_dataset_fid_cache(tmp_path):
    source = (Path(__file__).resolve().parents[1] / 'third_party/coral-lt-diffusion/main.py').read_text(
        encoding='utf-8')
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == 'resolve_fid_cache')
    flags = SimpleNamespace(data_type='cifar100lt', fid_cache='')
    namespace = {'FLAGS': flags, 'Path': Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<coral-main>', 'exec'), namespace)

    expected = tmp_path / 'cifar100.train.npz'
    expected.write_bytes(b'cache')
    flags.fid_cache = str(expected)
    assert namespace['resolve_fid_cache']() == str(expected.resolve())

    flags.fid_cache = 'stats/cifar100.train.npz'
    with pytest.raises(ValueError, match='absolute --fid_cache'):
        namespace['resolve_fid_cache']()

    wrong_dataset = tmp_path / 'cifar10.train.npz'
    wrong_dataset.write_bytes(b'cache')
    flags.fid_cache = str(wrong_dataset)
    with pytest.raises(ValueError, match='does not match data_type'):
        namespace['resolve_fid_cache']()
