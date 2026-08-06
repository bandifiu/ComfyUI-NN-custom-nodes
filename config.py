from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10
    
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "settings.toml"

@dataclass(slots=True)
class WildcardsConfig:
    path: str = "wildcards"
    cache: bool = True
    cleanup: bool = True
    delimiter: str = ", "
    max_iterations: int = 5

@dataclass(slots=True)
class ProfilerConfig:
    enable: bool = True
    console_log: bool = False
    console_log_threshold: float = 0.0
    precision: int = 2
    slow_time: float = 10.0
    active_color: str = "#4fc3f7"
    badge_color: str =  "#1e7832"
    badge_color_slow: str = "#962828"

def _update_section(section, cls, values):
    defaults = cls()

    for field in fields(cls):
        setattr(
            section,
            field.name,
            values.get(field.name, getattr(defaults, field.name))
        )

@dataclass(slots=True)
class Config:
    wildcards: WildcardsConfig = field(default_factory=WildcardsConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        if not path.exists():
            return cls()

        with path.open("rb") as f:
            data = tomllib.load(f)

        wildcards_data = data.get("wildcards", {})
        profiler_data = data.get("profiler", {})

        return cls(
            wildcards=WildcardsConfig(**wildcards_data),
            profiler=ProfilerConfig(**profiler_data)            
        )
    
    def update(self, path: Path = CONFIG_PATH) -> None:
        if not path.exists():
            return

        with path.open("rb") as f:
            data = tomllib.load(f)

        _update_section(
            self.wildcards,
            WildcardsConfig,
            data.get("wildcards", {})
        )

        _update_section(
            self.profiler,
            ProfilerConfig,
            data.get("profiler", {})
        )


config: Config = Config.load()
