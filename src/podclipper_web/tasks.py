"""Durable stage tasks with leases and child-inherited resource locks."""
from contextlib import contextmanager, ExitStack
import fcntl
import time

from .store import database

LEASE_SECONDS = 120


def enqueue(db, job_id, kind, reset=False):
    if kind not in ('prepare', 'render'):
        raise ValueError('Unknown web task kind')
    db.execute("INSERT OR IGNORE INTO tasks(job_id,kind,state) VALUES (?,?,'pending')", (job_id,kind))
    if reset:
        db.execute("UPDATE tasks SET state='pending',owner=NULL,lease_until=NULL WHERE job_id=? AND kind=?", (job_id,kind))
        if kind == 'prepare':
            db.execute("UPDATE tasks SET state='cancelled',owner=NULL,lease_until=NULL WHERE job_id=? AND kind='render'", (job_id,))


def renew(settings, owner):
    with database(settings) as db:
        db.execute("UPDATE tasks SET lease_until=? WHERE owner=? AND state='running'", (time.time()+LEASE_SECONDS,owner))


def _lock(stack, path):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    stream=stack.enter_context(path.open('a'))
    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return stream.fileno()


@contextmanager
def claim(settings, kind, owner):
    """Claim one task only when both its lease and its process locks allow it."""
    with ExitStack() as resource:
        fds=[]
        if kind == 'render':
            try:
                fds.append(_lock(resource,settings.data/'render.lock'))
            except BlockingIOError:
                yield None
                return
        with database(settings) as db:
            rows=db.execute("SELECT t.job_id FROM tasks t JOIN jobs j ON j.id=t.job_id "
                            "WHERE t.kind=? AND j.state IN ('queued','running') AND "
                            "(t.state='pending' OR (t.state='running' AND t.lease_until<?)) "
                            "ORDER BY j.created_at,j.id LIMIT 20", (kind,time.time())).fetchall()
        for row in rows:
            with ExitStack() as job_lock:
                try:
                    fd=_lock(job_lock,settings.data/'jobs'/row['job_id']/'process.lock')
                except BlockingIOError:
                    continue
                with database(settings) as db:
                    db.execute('BEGIN IMMEDIATE')
                    changed=db.execute("UPDATE tasks SET state='running',owner=?,lease_until=?,attempts=attempts+1 "
                                       "WHERE job_id=? AND kind=? AND (state='pending' OR (state='running' AND lease_until<?))",
                                       (owner,time.time()+LEASE_SECONDS,row['job_id'],kind,time.time()))
                    if not changed.rowcount:
                        continue
                    job=dict(db.execute('SELECT * FROM jobs WHERE id=?',(row['job_id'],)).fetchone())
                    db.execute("UPDATE jobs SET state='running',heartbeat=?,updated_at=?,attempts=attempts+? WHERE id=?", (time.time(),time.time(),int(kind=='prepare'),job['id']))
                try:
                    yield {**job,'task_kind':kind,'task_owner':owner,'lock_fds':tuple([fd,*fds])}
                except BaseException:
                    with database(settings) as db:
                        db.execute("UPDATE tasks SET owner=NULL,lease_until=0 WHERE job_id=? AND kind=? AND owner=? AND state='running'",
                                   (job['id'],kind,owner))
                    raise
                return
        yield None


def complete(settings, task, next_kind=None):
    with database(settings) as db:
        db.execute('BEGIN IMMEDIATE')
        changed=db.execute("UPDATE tasks SET state='done',lease_until=NULL WHERE job_id=? AND kind=? AND owner=? AND state='running'",
                           (task['id'],task['task_kind'],task['task_owner']))
        if changed.rowcount and next_kind:
            enqueue(db,task['id'],next_kind,reset=True)
            db.execute("UPDATE jobs SET state='queued',stage='queued',updated_at=? WHERE id=?",(time.time(),task['id']))
