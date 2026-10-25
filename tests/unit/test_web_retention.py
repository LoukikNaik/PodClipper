"""The configured owner's finished reels survive normal result expiration."""
import time
from podclipper_web.store import Settings, database, initialize
from podclipper_web.worker import expire


def test_expiration_keeps_owner_reels_but_removes_other_accounts_results(tmp_path):
    settings = Settings(tmp_path, 'test.apps.googleusercontent.com', 'https://podclipper.loukik.dev', True, 10800, 7, 'owner-sub')
    initialize(settings)
    old = time.time() - 30 * 86400
    with database(settings) as db:
        for owner in ('owner-sub', 'other-sub'):
            db.execute('INSERT INTO users VALUES (?,?,?,?)', (owner, 'same@example.com', owner, old))
            db.execute("INSERT INTO jobs (id,user_id,video_id,url,clips,quota_month,request_key,created_at,updated_at,finished_at,state,stage) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (owner, owner, 'abcdefghijk', 'https://youtu.be/abcdefghijk', 1, '2026-09', owner, old, old, old, 'completed', 'completed'))
            db.execute('INSERT INTO reels (id,job_id,title,path,duration,created_at) VALUES (?,?,?,?,?,?)', (owner, owner, 'A reel', 'reels/one.mp4', 30, old))
            directory = tmp_path / 'jobs' / owner / 'reels'
            directory.mkdir(parents=True)
            (directory / 'one.mp4').write_bytes(b'keep me')
    expire(settings)
    with database(settings) as db:
        assert db.execute('SELECT state FROM jobs WHERE id=?', ('owner-sub',)).fetchone()[0] == 'completed'
        assert db.execute('SELECT count(*) FROM reels WHERE job_id=?', ('owner-sub',)).fetchone()[0] == 1
        assert db.execute('SELECT state FROM jobs WHERE id=?', ('other-sub',)).fetchone()[0] == 'expired'
        assert db.execute('SELECT count(*) FROM reels WHERE job_id=?', ('other-sub',)).fetchone()[0] == 0
    assert (tmp_path / 'jobs/owner-sub/reels/one.mp4').read_bytes() == b'keep me'
    assert not (tmp_path / 'jobs/other-sub').exists()
