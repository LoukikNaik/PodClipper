"""Validated access groups keyed by verified Google account identifiers."""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AccessGroup:
    monthly_videos: int | None
    retention_days: int | None
    admin: bool = False


@dataclass(frozen=True)
class AccessPolicy:
    default_group: str
    groups: dict[str, AccessGroup]
    users: dict[str, str]

    def for_user(self, user_id):
        return self.groups[self.users.get(user_id, self.default_group)]


def load_policy(path: Path):
    try:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict) or set(data) != {'default_group', 'groups', 'users'}:
            raise ValueError('Expected default_group, groups, and users')
        groups = {}
        if not isinstance(data['groups'], dict) or not isinstance(data['users'], dict):
            raise ValueError('Groups and users must be mappings')
        for name, value in data['groups'].items():
            if not isinstance(name, str) or not name or not isinstance(value, dict) or set(value) != {'monthly_videos', 'retention_days', 'admin'}:
                raise ValueError('Every group needs monthly_videos, retention_days, and admin')
            for key in ('monthly_videos', 'retention_days'):
                limit = value[key]
                if limit is not None and (type(limit) is not int or limit < 1):
                    raise ValueError(f'{key} must be a positive integer or null')
            if type(value['admin']) is not bool:
                raise ValueError('Admin must be true or false')
            groups[name] = AccessGroup(**value)
        default = data['default_group']
        if not isinstance(default, str) or default not in groups or groups[default].admin:
            raise ValueError('Default group must exist and cannot grant admin')
        for uid, group in data['users'].items():
            if not isinstance(uid, str) or not uid.strip() or not isinstance(group, str) or group not in groups:
                raise ValueError('User IDs must be quoted strings assigned to existing groups')
        return AccessPolicy(default, groups, dict(data['users']))
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid web access configuration: {exc}') from exc
