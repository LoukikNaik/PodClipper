"""Group policies separate membership allowances from administrative authority."""
import pytest
from podclipper_web.store import Settings


def settings(tmp_path):
    return Settings(tmp_path,'test.apps.googleusercontent.com','https://example.com',True,10800,7,'legacy-owner')


def test_membership_has_higher_limits_without_admin_access(tmp_path):
    from podclipper_web.access import load_policy
    s=settings(tmp_path)
    path=tmp_path/'access.yaml'
    path.write_text('''default_group: free
groups:
  free: {monthly_videos: 5, retention_days: 7, admin: false}
  member: {monthly_videos: 50, retention_days: 30, admin: false}
  owner: {monthly_videos: null, retention_days: null, admin: true}
users:
  paid-sub: member
  owner-sub: owner
''')
    s.access_policy=load_policy(path)
    assert s.access_for('paid-sub').monthly_videos==50
    assert not s.access_for('paid-sub').admin
    assert s.access_for('owner-sub').admin
    assert s.access_for('owner-sub').retention_days is None
    assert s.access_for('unknown').monthly_videos==5
    assert not s.access_for('legacy-owner').admin


def test_bad_membership_config_fails_closed(tmp_path):
    from podclipper_web.access import load_policy
    p=tmp_path/'access.yaml';p.write_text('default_group: free\ngroups: {}\nusers: {}\n')
    with pytest.raises(ValueError):load_policy(p)
