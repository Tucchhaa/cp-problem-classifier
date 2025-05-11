from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ProblemPreview:
    id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    quality: Optional[str] = None
    thumbs_up: Optional[str] = None
    thumbs_down: Optional[str] = None
    user_count: Optional[str] = None
    acceptance_rate: Optional[str] = None

@dataclass
class Problem:
    id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    quality: Optional[str] = None
    thumbs_up: Optional[str] = None
    thumbs_down: Optional[str] = None
    user_count: Optional[str] = None
    acceptance_rate: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""
    task_description: str = ""
    input_format: str = ""
    output_format: str = ""
    example: str = ""
    author: str = ""
    author_url: str = ""
    date: Optional[str] = None
    time_limit: Optional[str] = None
    source_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    cluster: Optional[str] = None
    languages: Optional[str] = None
    resource: Optional[str] = None
