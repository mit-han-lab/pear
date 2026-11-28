from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

from pear.agent_loop.episode import Episode

from .reporting import run_metadata, write_report


async def preflight(config, runtime, dataset, timeout=900):
    root = Path(config.run_dir) / "preflight"
    report = {**run_metadata(config), "resources": runtime.resource_plan, "families": {}, "status": "running"}
    write_report(root / "report.json", report)

    def cameras(name, stage, session, result):
        images = result["images"]
        if not images or len(images) != len(session.env_class.CAMERAS):
            raise ValueError(f"{name} {stage} did not render every camera")
        frames = []
        for camera, image in zip(session.env_class.CAMERAS, images, strict=True):
            if min(image.size) <= 0:
                raise ValueError(f"{name} {camera} returned an empty image")
            path = root / name.rsplit(".", 1)[-1] / f"{stage}_{camera}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            frames.append(
                {
                    "camera": camera,
                    "size": list(image.size),
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        return frames

    try:
        for name, session in runtime.sessions.items():
            task = next((task for task in dataset.tasks if task.env_class == name), None)
            if task is None:
                raise ValueError(f"preflight needs an episode for {name}")
            episode, started = Episode(dict(task.env_args)), time.monotonic()
            try:
                async with asyncio.timeout(timeout):
                    _, initial = await session.open(episode)
                    initial_frames = cameras(name, "reset", session, initial)
                    action = {
                        part: {quantity: 0.0 for quantity in quantities}
                        for part, quantities in session.env_class.SCHEMA.items()
                    }
                    _, result = await session.step(episode, action)
                    episode.update(result)
                    if not result["actions"]:
                        raise ValueError(f"{name} executed no simulator action")
                    final_frames = cameras(name, "action", session, result)
                    report["families"][name] = {
                        "task": task.env_args,
                        "action": action,
                        "native_steps": len(result["actions"]),
                        "reset_cameras": initial_frames,
                        "action_cameras": final_frames,
                        "seconds": time.monotonic() - started,
                    }
                    write_report(root / "report.json", report)
            finally:
                await session.close(episode)
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(root / "report.json", report)
