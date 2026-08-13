from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import os
import tempfile


@dataclass(frozen=True)
class BasePathsA:
    project_root: Path
    scenarios_a_dir: Path
    results_a_root: Path
    sim_a_dir: Path
    modules_dir: Path
    runs_scenarios_dir: Path
    runs_results_dir: Path
    latest_run_file: Path


@dataclass(frozen=True)
class RunPathsA:
    run_id: str
    run_scenarios_dir: Path
    config_dir: Path
    traj_dir: Path
    buildings_dir: Path
    tunnel_dir: Path
    run_results_dir: Path
    raw_dir: Path
    tables_dir: Path
    figures_dir: Path
    manifest_path: Path


def _detect_project_root(from_file: Path) -> Path:
    # Release layout: <repo>/py/paths_C.py.  Keep generated scenarios and
    # results inside the cloned repository instead of assuming the historical
    # PE_V2X_Reliability parent-directory layout.
    return from_file.resolve().parents[1]


def _detect_variant(from_file: Path) -> str:
    return 'C'


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)




def default_run_prefix() -> str:
    this_file = Path(__file__)
    variant = _detect_variant(this_file)
    return f"{variant}_"

def get_base_paths_a() -> BasePathsA:
    this_file = Path(__file__)
    root = _detect_project_root(this_file)
    variant = _detect_variant(this_file)

    scenarios_a_dir = root / 'workspace' / 'scenarios'
    results_a_root = root / 'workspace' / 'results'
    sim_a_dir = root / 'py'
    modules_dir = sim_a_dir / 'modules'
    runs_scenarios_dir = scenarios_a_dir / 'runs'
    runs_results_dir = results_a_root / 'runs'
    latest_run_file = results_a_root / 'LATEST_RUN.json'

    return BasePathsA(
        project_root=root,
        scenarios_a_dir=scenarios_a_dir,
        results_a_root=results_a_root,
        sim_a_dir=sim_a_dir,
        modules_dir=modules_dir,
        runs_scenarios_dir=runs_scenarios_dir,
        runs_results_dir=runs_results_dir,
        latest_run_file=latest_run_file,
    )


def make_run_id(prefix: str = '') -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}{ts}" if prefix else ts


def ensure_base_dirs_a() -> BasePathsA:
    b = get_base_paths_a()
    b.modules_dir.mkdir(parents=True, exist_ok=True)
    b.runs_scenarios_dir.mkdir(parents=True, exist_ok=True)
    b.runs_results_dir.mkdir(parents=True, exist_ok=True)
    return b


def ensure_run_dirs_a(run_id: str, save_as_latest: bool = True, meta: dict | None = None) -> RunPathsA:
    b = ensure_base_dirs_a()
    run_scenarios_dir = b.runs_scenarios_dir / run_id
    config_dir = run_scenarios_dir / 'config'
    traj_dir = run_scenarios_dir / 'trajectories'
    buildings_dir = run_scenarios_dir / 'buildings'
    tunnel_dir = run_scenarios_dir / 'tunnel'
    run_results_dir = b.runs_results_dir / run_id
    raw_dir = run_results_dir / 'raw'
    tables_dir = run_results_dir / 'tables'
    figures_dir = run_results_dir / 'figures'
    for d in [config_dir, traj_dir, buildings_dir, tunnel_dir, raw_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)
    manifest_path = run_results_dir / 'run_manifest.json'
    now = datetime.now().isoformat(timespec='seconds')
    if manifest_path.exists():
        try:
            meta2 = json.loads(manifest_path.read_text(encoding='utf-8'))
            if not isinstance(meta2, dict):
                meta2 = {}
        except Exception:
            meta2 = {}
    else:
        meta2 = {}
    meta2.setdefault('run_id', run_id)
    meta2.setdefault('created_at', now)
    stage_meta = dict(meta or {})
    if stage_meta:
        stage_entry = {'at': now, **stage_meta}
        stages = meta2.setdefault('stages', [])
        if not isinstance(stages, list):
            stages = []
            meta2['stages'] = stages
        stages.append(stage_entry)
        meta2['last_stage'] = stage_meta.get('script', 'unknown')
    _atomic_write_json(manifest_path, meta2)
    if save_as_latest:
        _atomic_write_json(b.latest_run_file, {'latest_run_id': run_id, 'updated_at': now})
    return RunPathsA(run_id, run_scenarios_dir, config_dir, traj_dir, buildings_dir, tunnel_dir,
                     run_results_dir, raw_dir, tables_dir, figures_dir, manifest_path)


def load_latest_run_id() -> str | None:
    b = get_base_paths_a()
    if not b.latest_run_file.exists():
        return None
    try:
        obj = json.loads(b.latest_run_file.read_text(encoding='utf-8'))
        return obj.get('latest_run_id')
    except Exception:
        return None
