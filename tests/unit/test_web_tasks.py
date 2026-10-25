"""Durable task claims must not steal active or still-running child work."""
import time
from podclipper_web.store import Settings, database, initialize
from podclipper_web.tasks import enqueue, claim, complete


def test_task_claim_is_exclusive_and_completing_preparation_queues_render(tmp_path):
    settings=Settings(tmp_path,'test.apps.googleusercontent.com','https://podclipper.loukik.dev',True,10800,7)
    initialize(settings)
    with database(settings) as db:
        db.execute("INSERT INTO users VALUES ('owner','owner@example.com','Owner',0)")
        db.execute("INSERT INTO jobs (id,user_id,video_id,url,clips,quota_month,request_key,created_at,updated_at) VALUES ('job','owner','abcdefghijk','https://youtu.be/abcdefghijk',0,'2026-10','request',0,0)")
        enqueue(db,'job','prepare')
    with claim(settings,'prepare','worker-a') as task:
        assert task['id']=='job'
        with claim(settings,'prepare','worker-b') as other:
            assert other is None
        complete(settings,task,next_kind='render')
    with claim(settings,'render','worker-b') as task:
        assert task['id']=='job'
        complete(settings,task)
    with claim(settings,'render','worker-a') as other:
        assert other is None


def test_expired_lease_cannot_steal_a_live_process_lock(tmp_path):
    settings=Settings(tmp_path,'test.apps.googleusercontent.com','https://podclipper.loukik.dev',True,10800,7)
    initialize(settings)
    with database(settings) as db:
        db.execute("INSERT INTO users VALUES ('owner','owner@example.com','Owner',0)")
        db.execute("INSERT INTO jobs (id,user_id,video_id,url,clips,quota_month,request_key,created_at,updated_at) VALUES ('job','owner','abcdefghijk','https://youtu.be/abcdefghijk',0,'2026-10','request',0,0)")
        enqueue(db,'job','prepare')
    with claim(settings,'prepare','old') as task:
        with database(settings) as db:
            db.execute('UPDATE tasks SET lease_until=0')
        with claim(settings,'prepare','new') as other:
            assert other is None
    with claim(settings,'prepare','new') as task:
        assert task['task_owner']=='new'


def test_two_preparations_can_overlap_but_rendering_has_one_global_slot(tmp_path):
    settings=Settings(tmp_path,'test.apps.googleusercontent.com','https://podclipper.loukik.dev',True,10800,7)
    initialize(settings)
    with database(settings) as db:
        db.execute("INSERT INTO users VALUES ('owner','owner@example.com','Owner',0)")
        for i in range(3):
            db.execute("INSERT INTO jobs (id,user_id,video_id,url,clips,quota_month,request_key,created_at,updated_at) VALUES (?,'owner','abcdefghijk','https://youtu.be/abcdefghijk',0,'2026-10',?,0,0)",(str(i),str(i)))
            enqueue(db,str(i),'prepare' if i<2 else 'render')
    with claim(settings,'prepare','one') as a,claim(settings,'prepare','two') as b,claim(settings,'render','three') as c:
        assert len({a['id'],b['id'],c['id']})==3
        with database(settings) as db:
            enqueue(db,a['id'],'render')
        with claim(settings,'render','four') as other:
            assert other is None


def test_failed_runner_releases_its_lease_instead_of_renewing_an_orphan(tmp_path):
    import pytest
    from podclipper_web.tasks import renew
    settings=Settings(tmp_path,'test.apps.googleusercontent.com','https://podclipper.loukik.dev',True,10800,7)
    initialize(settings)
    with database(settings) as db:
        db.execute("INSERT INTO users VALUES ('owner','owner@example.com','Owner',0)")
        db.execute("INSERT INTO jobs (id,user_id,video_id,url,clips,quota_month,request_key,created_at,updated_at) VALUES ('job','owner','abcdefghijk','https://youtu.be/abcdefghijk',0,'2026-10','request',0,0)")
        enqueue(db,'job','prepare')
    with pytest.raises(RuntimeError):
        with claim(settings,'prepare','old'):
            raise RuntimeError('runner failed')
    renew(settings,'old')
    with claim(settings,'prepare','new') as task:
        assert task is not None


def test_render_child_retains_the_slot_after_parent_releases_its_claim(tmp_path):
    import subprocess,sys
    settings=Settings(tmp_path,'test.apps.googleusercontent.com','https://podclipper.loukik.dev',True,10800,7)
    initialize(settings)
    with database(settings) as db:
        db.execute("INSERT INTO users VALUES ('owner','owner@example.com','Owner',0)")
        db.execute("INSERT INTO jobs (id,user_id,video_id,url,clips,quota_month,request_key,created_at,updated_at) VALUES ('job','owner','abcdefghijk','https://youtu.be/abcdefghijk',0,'2026-10','request',0,0)")
        enqueue(db,'job','render')
    child=None
    try:
        with claim(settings,'render','old') as task:
            child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],pass_fds=task['lock_fds'])
            with database(settings) as db:
                db.execute('UPDATE tasks SET lease_until=0')
        with claim(settings,'render','new') as other:
            assert other is None
        child.terminate();child.wait(timeout=3)
        with claim(settings,'render','new') as other:
            assert other is not None
    finally:
        if child is not None and child.poll() is None:
            child.kill();child.wait()
