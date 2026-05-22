from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HYDRA_DIR = REPO_ROOT / "scripts" / "hydra"
CONFIG_PATTERN = re.compile(r"configs/[A-Za-z0-9_.-]+\.yaml")
LEGACY_CLUSTER_ROOT_PATTERN = re.compile(r"/(?:data|scratch)/brussel/112/vsc11249")
SHARED_PATH_POLICY_SCRIPTS = {
    "download_mimic.sh",
    "download_physionet.sh",
    "evaluate_checkpoint.sh",
    "explain_checkpoint.sh",
    "prepare_mimic_jpg.sh",
    "preprocess_dicom.sh",
    "run_phase1.sh",
    "setup_clip.sh",
    "setup_env.sh",
    "setup_env_sbatch.sh",
    "train_baseline.sh",
    "train_neurosymbolic.sh",
    "umls_download.sh",
}


def test_hydra_scripts_reference_existing_configs() -> None:
    missing: list[str] = []

    for script_path in HYDRA_DIR.glob("*.sh"):
        contents = script_path.read_text(encoding="utf-8")
        for rel_path in sorted(set(CONFIG_PATTERN.findall(contents))):
            if not (REPO_ROOT / rel_path).exists():
                missing.append(f"{script_path.name}: {rel_path}")

    assert not missing, "Missing Hydra config references:\n" + "\n".join(missing)


def test_hydra_scripts_are_bash_parseable() -> None:
    for script_path in HYDRA_DIR.glob("*.sh"):
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"bash -n failed for {script_path.name}: {result.stderr}"


def test_hydra_scripts_source_shared_path_policy() -> None:
    missing: list[str] = []

    for script_name in sorted(SHARED_PATH_POLICY_SCRIPTS):
        script_path = HYDRA_DIR / script_name
        contents = script_path.read_text(encoding="utf-8")
        if "common_paths.sh" not in contents:
            missing.append(script_name)

    assert not missing, "Hydra scripts missing shared path policy:\n" + "\n".join(missing)


def test_hydra_surfaces_avoid_hardcoded_legacy_cluster_roots() -> None:
    targets = [*HYDRA_DIR.glob("*.sh"), REPO_ROOT / "configs" / "phase1_kg_gpu.yaml"]
    offenders: list[str] = []

    for target in targets:
        contents = target.read_text(encoding="utf-8")
        if LEGACY_CLUSTER_ROOT_PATTERN.search(contents):
            offenders.append(target.relative_to(REPO_ROOT).as_posix())

    assert not offenders, "Hardcoded legacy cluster roots remain in:\n" + "\n".join(offenders)