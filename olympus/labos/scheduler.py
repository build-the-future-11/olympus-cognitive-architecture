from __future__ import annotations

from dataclasses import dataclass

from olympus.labos.manifest import ProjectManifest, ResourceProfile


@dataclass(slots=True)
class ScheduledTask:
    project_id: str
    command: str
    priority: int
    profile: ResourceProfile


class ResourceAwareScheduler:
    def schedule(
        self,
        manifests: list[ProjectManifest],
        profile: ResourceProfile,
    ) -> list[ScheduledTask]:
        tasks: list[ScheduledTask] = []
        for manifest in manifests:
            for entry in manifest.entry_points:
                if entry.profile != profile:
                    continue
                priority = 0
                if manifest.manifest_source.value == "declared":
                    priority += 2
                if manifest.project_id == "olympus":
                    priority += 3
                if manifest.resource_requirements.gpu_optional:
                    priority -= 1
                tasks.append(
                    ScheduledTask(
                        project_id=manifest.project_id,
                        command=entry.command,
                        priority=priority,
                        profile=profile,
                    )
                )
        return sorted(tasks, key=lambda item: item.priority, reverse=True)
