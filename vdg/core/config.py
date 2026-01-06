"""
Configuration management for the VDG framework.

Provides a flexible configuration system supporting JSON files
and environment variables.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class GlobalSettings:
    """Global pipeline settings."""
    prefix: str = ""
    vdg_path: str = "~/bin/vdg"
    keep_temp_files: bool = False
    post_process_function: str = "proc00"


@dataclass
class PassConfig:
    """Configuration for a single processing pass."""
    name: str = "stab"
    type: str = "track"  # "track" or "perspective"
    track_files: list[str] = field(default_factory=list)
    persp_file: str = ""
    stab_options: str = "-wo -wm"
    frameavg_options: str = "-wa -um -tk umavg"
    do_stab: bool = True
    do_frameavg: bool = True
    do_postProcess: bool = True
    do_metadata_cp: bool = True


@dataclass
class MovieConfig:
    """Configuration for a single movie's processing."""
    passes: list[PassConfig] = field(default_factory=list)


@dataclass
class Config:
    """
    Main configuration container.
    
    Example:
        config = Config.load("process_config.json")
        for movie_id, movie_config in config.movies.items():
            for pass_cfg in movie_config.passes:
                process(movie_id, pass_cfg)
    """
    global_settings: GlobalSettings = field(default_factory=GlobalSettings)
    movies: dict[str, MovieConfig] = field(default_factory=dict)
    
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load configuration from a JSON file."""
        return load_config(path)
    
    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        save_config(self, path)
    
    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            "global_settings": asdict(self.global_settings),
            "movies": {
                movie_id: {
                    "passes": [asdict(p) for p in mc.passes]
                }
                for movie_id, mc in self.movies.items()
            },
        }


def load_config(path: str | Path) -> Config:
    """
    Load configuration from a JSON file.
    
    Args:
        path: Path to the JSON configuration file
        
    Returns:
        Parsed Config object
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the JSON is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    # Parse global settings
    gs_data = data.get("global_settings", {})
    global_settings = GlobalSettings(
        prefix=gs_data.get("prefix", ""),
        vdg_path=gs_data.get("vdg_path", "~/bin/vdg"),
        keep_temp_files=gs_data.get("keep_temp_files", False),
        post_process_function=gs_data.get("post_process_function", "proc00"),
    )
    
    # Parse movies
    movies = {}
    for movie_id, movie_data in data.get("movies", {}).items():
        passes = []
        for pass_data in movie_data.get("passes", []):
            passes.append(PassConfig(
                name=pass_data.get("name", "stab"),
                type=pass_data.get("type", "track"),
                track_files=pass_data.get("track_files", []),
                persp_file=pass_data.get("persp_file", ""),
                stab_options=pass_data.get("stab_options", "-wo -wm"),
                frameavg_options=pass_data.get("frameavg_options", "-wa -um -tk umavg"),
                do_stab=pass_data.get("do_stab", True),
                do_frameavg=pass_data.get("do_frameavg", True),
                do_postProcess=pass_data.get("do_postProcess", True),
                do_metadata_cp=pass_data.get("do_metadata_cp", True),
            ))
        movies[movie_id] = MovieConfig(passes=passes)
    
    return Config(global_settings=global_settings, movies=movies)


def save_config(config: Config, path: str | Path) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration object to save
        path: Output path for the JSON file
    """
    path = Path(path)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def create_example_config(path: str | Path = "process_config.json") -> Config:
    """
    Create an example configuration file.
    
    Args:
        path: Output path for the example config
        
    Returns:
        The created Config object
    """
    parent_dir = os.path.basename(os.getcwd())
    
    config = Config(
        global_settings=GlobalSettings(
            prefix=parent_dir,
            vdg_path="~/bin/vdg",
            keep_temp_files=False,
            post_process_function="proc00",
        ),
        movies={
            "example_clip_001": MovieConfig(passes=[
                PassConfig(
                    name="stab",
                    type="track",
                    track_files=["track01.crv", "track02.crv"],
                    stab_options="-wo -wm -sw -yf -xp 3000 -yp 2000",
                    frameavg_options="-wa -um -tk umavg",
                    do_stab=True,
                    do_frameavg=True,
                    do_postProcess=True,
                    do_metadata_cp=True,
                ),
            ]),
        },
    )
    
    config.save(path)
    print(f"Created example configuration: {path}")
    return config


def get_env_config(prefix: str = "VDG_") -> dict[str, Any]:
    """
    Get configuration from environment variables.
    
    All environment variables starting with the prefix will be included.
    Variable names are converted to lowercase with the prefix removed.
    
    Example:
        VDG_VERBOSE=true -> {"verbose": "true"}
    """
    config = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            config[config_key] = value
    return config
