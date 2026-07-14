#!/usr/bin/env python3
"""Non-destructive, resumable SalientGS paper reproduction runner."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCENE_GROUPS = {
    "mip360": [
        "bicycle", "bonsai", "counter", "flowers", "garden",
        "kitchen", "room", "stump", "treehill",
    ],
    "deep_blending": ["drjohnson", "playroom"],
    "tanks_temples": ["train", "truck"],
}
ALL_SCENES = [scene for scenes in SCENE_GROUPS.values() for scene in scenes]
PROFILES = {
    "smoke": (["stump"], 1_000),
    "representative": (["garden", "counter", "playroom", "train"], 7_000),
    "full": (ALL_SCENES, 30_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("reproduction_runs"))
    parser.add_argument("--profile", choices=PROFILES, default="representative")
    parser.add_argument("--scenes", nargs="+", choices=ALL_SCENES)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap-max", type=int, default=1_500_000)
    parser.add_argument("--method", choices=("guided", "vanilla"), default="guided")
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument(
        "--stages", nargs="+", choices=("features", "sfm", "train"),
        default=["features", "sfm", "train"],
    )
    parser.add_argument(
        "--reference-sfm", action="store_true",
        help="Use source sparse/ only for isolated GS tuning; skips features/SfM.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun requested stages.")
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def stage_complete(
    scene_dir: Path, stage: str, result_name: str = "gsplat", max_steps: int | None = None
) -> bool:
    if stage == "features":
        return (scene_dir / "database.db").is_file()
    if stage == "sfm":
        model = scene_dir / "sparse" / "0"
        return all((model / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))
    stats_dir = scene_dir / result_name / "stats"
    if max_steps is not None:
        return (stats_dir / f"val_step{max_steps - 1:04d}.json").is_file()
    return any(stats_dir.glob("val_step*.json"))


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        header = f"\n$ {' '.join(command)}\n"
        print(header, end="", flush=True)
        log.write(header)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed


def prepare_scene(source: Path, target: Path, reference_sfm: bool) -> None:
    images = source / "images"
    if not images.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images}")
    target.mkdir(parents=True, exist_ok=True)
    image_link = target / "images"
    if not image_link.exists():
        image_link.symlink_to(images.resolve(), target_is_directory=True)
    if reference_sfm and not (target / "sparse").exists():
        sparse = source / "sparse"
        if not sparse.is_dir():
            raise FileNotFoundError(f"Missing reference SfM model: {sparse}")
        (target / "sparse").symlink_to(sparse.resolve(), target_is_directory=True)


def read_metrics(scene_dir: Path, result_name: str) -> dict[str, object] | None:
    files = sorted((scene_dir / result_name / "stats").glob("val_step*.json"))
    if not files:
        return None
    with files[-1].open(encoding="utf-8") as handle:
        result = json.load(handle)
    result["stats_file"] = str(files[-1])
    return result


def write_summary(rows: list[dict[str, object]], output_dir: Path, method: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"summary_{method}.json").write_text(json.dumps(rows, indent=2) + "\n")
    fieldnames = ["scene", "method", "seed", "max_steps", "psnr", "ssim", "lpips", "num_GS", "total_seconds"]
    with (output_dir / f"summary_{method}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    profile_scenes, profile_steps = PROFILES[args.profile]
    scenes = args.scenes or profile_scenes
    max_steps = args.max_steps or profile_steps
    output_dir = args.work_root.resolve() / args.profile
    result_name = "gsplat" if args.method == "guided" else "gsplat_vanilla"
    trainer_config = "mcmc_importance" if args.method == "guided" else "mcmc"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")
    # The repository's fastmap/ project directory has the same name as its
    # inner Python package. Put the package root first to avoid namespace
    # shadowing when this runner is launched from the repository root.
    python_paths = [str(repo / "fastmap"), str(repo)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    summary_json = output_dir / f"summary_{args.method}.json"
    if summary_json.is_file():
        rows = json.loads(summary_json.read_text(encoding="utf-8"))
    else:
        rows: list[dict[str, object]] = []

    for scene in scenes:
        source = dataset_root / scene
        scene_dir = output_dir / f"{scene}_seed{args.seed}"
        prepare_scene(source, scene_dir, args.reference_sfm)
        timings: dict[str, float] = {}
        requested = [] if args.reference_sfm else args.stages

        if "features" in requested and (args.force or not stage_complete(scene_dir, "features")):
            if args.force:
                for path in (scene_dir / "database.db", scene_dir / "pairs.txt"):
                    path.unlink(missing_ok=True)
            timings["features"] = run_logged([
                sys.executable, "-m", "global_corr.global_matching",
                "--image_dir", str(scene_dir / "images"),
                "--output_dir", str(scene_dir),
                "--k_neighbors", "20", "--top_k_pairs", "20",
                "--gmm_components", "64", "--gmm_seed", str(args.seed),
            ], scene_dir / "logs" / "features.log", env)

        if "sfm" in requested and (args.force or not stage_complete(scene_dir, "sfm")):
            if not stage_complete(scene_dir, "features"):
                raise RuntimeError(f"Feature stage incomplete for {scene}")
            if args.force:
                sparse = scene_dir / "sparse"
                if sparse.is_symlink():
                    sparse.unlink()
                elif sparse.exists():
                    shutil.rmtree(sparse)
            timings["sfm"] = run_logged([
                sys.executable, "-m", "salientgs.scripts.sfm", "--headless",
                "--database", str(scene_dir / "database.db"),
                "--image_dir", str(scene_dir / "images"),
                "--output_dir", str(scene_dir),
            ], scene_dir / "logs" / "sfm.log", env)

        if "train" in args.stages and (
            args.force
            or not stage_complete(scene_dir, "train", result_name, max_steps)
        ):
            if not stage_complete(scene_dir, "sfm"):
                raise RuntimeError(f"SfM stage incomplete for {scene}")
            result_dir = scene_dir / result_name
            if result_dir.exists():
                shutil.rmtree(result_dir)
            timings["train"] = run_logged([
                sys.executable, str(repo / "salientgs" / "vis" / "gsplat_joint.py"),
                trainer_config, "--data-dir", str(scene_dir),
                "--image-folder-name", "images", "--data-factor", str(args.data_factor),
                "--result-dir", str(result_dir), "--max-steps", str(max_steps),
                "--seed", str(args.seed), "--strategy.cap-max", str(args.cap_max),
                "--disable-viewer", "--no-use-wandb", "--eval-epochs", "1.0",
                "--save-epochs", "1.0",
            ], scene_dir / "logs" / "train.log", env)

        metrics = read_metrics(scene_dir, result_name) or {}
        previous = next(
            (existing for existing in rows if existing.get("scene") == scene), None
        )
        elapsed = sum(timings.values())
        if not timings and previous is not None:
            elapsed = float(previous.get("total_seconds", 0.0))
        row: dict[str, object] = {
            "scene": scene, "method": args.method, "seed": args.seed, "max_steps": max_steps,
            "total_seconds": elapsed, **metrics,
        }
        rows = [existing for existing in rows if existing.get("scene") != scene]
        rows.append(row)
        rows.sort(key=lambda item: ALL_SCENES.index(str(item["scene"])))
        (scene_dir / f"run_{args.method}.json").write_text(json.dumps({
            "profile": args.profile, "dataset_source": str(source.resolve()),
            "reference_sfm": args.reference_sfm, "timings_seconds": timings,
            **row,
        }, indent=2) + "\n")
        write_summary(rows, output_dir, args.method)

    print(f"Summary: {output_dir / f'summary_{args.method}.csv'}")


if __name__ == "__main__":
    main()
