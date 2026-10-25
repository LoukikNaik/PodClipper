"""Owned, bounded parallel uploads enter the normal queue only after validation."""
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient
from podclipper_web.api import create_app
from podclipper_web.store import Settings, database, initialize, usage


@pytest.fixture
def upload_client(tmp_path):
    settings = Settings(tmp_path.resolve(), 'test.apps.googleusercontent.com', 'https://podclipper.loukik.dev', True, 10800, 7)
    initialize(settings)
    with database(settings) as db:
        for uid in ('owner', 'other'):
            db.execute('INSERT INTO users VALUES (?,?,?,?)', (uid, uid+'@example.com', uid, time.time()))
            db.execute('INSERT INTO sessions VALUES (?,?,?,?)',
                       (hashlib.sha256(uid.encode()).hexdigest(), uid, 'csrf', time.time()+1000))
    with patch.object(Settings, 'load', return_value=settings), TestClient(create_app(), base_url='https://api.example.com') as client:
        client.cookies.set(settings.cookie, 'owner')
        yield client, settings


def headers():
    return {'Origin': 'https://podclipper.loukik.dev', 'X-CSRF-Token': 'csrf'}


def begin(client, size=40000, key='upload-request-0000001'):
    result=client.post('/uploads', json={'filename':'Podcast.mp4','size':size,'request_key':key}, headers=headers())
    assert result.status_code == 200, result.text
    return result.json()


def test_upload_creation_requires_login_csrf_and_strict_sub_1gb_limit(upload_client):
    client, settings=upload_client
    client.cookies.clear()
    assert client.post('/uploads',json={'filename':'a.mp4','size':1,'request_key':'upload-request-0000001'},headers=headers()).status_code==401
    client.cookies.set(settings.cookie,'owner')
    assert client.post('/uploads',json={'filename':'a.mp4','size':1,'request_key':'upload-request-0000001'},headers={'Origin':settings.origin}).status_code==403
    assert client.post('/uploads',json={'filename':'a.mp4','size':1_000_000_000,'request_key':'upload-request-0000001'},headers=headers()).status_code==422
    job=begin(client)
    assert job['state']=='uploading' and job['source_type']=='upload'
    assert job['upload']['chunk_bytes']==32_000_000
    assert job['upload']['parallel_requests']==4
    assert client.get('/config').json()['max_upload_bytes']==1_000_000_000
    assert begin(client)['id']==job['id']
    with database(settings) as db: assert usage(db,'owner')==1


def test_chunks_can_arrive_out_of_order_replay_safely_and_finalize_once(upload_client,monkeypatch):
    import podclipper_web.uploads as uploads
    monkeypatch.setattr(uploads,'CHUNK_BYTES',20000)
    client,settings=upload_client; job=begin(client); jid=job['id']
    body0=b'a'*20000;body1=b'b'*20000
    url=f'/uploads/{jid}/chunks/'
    assert client.post(url+'1',content=body1,headers=headers()).status_code==200
    assert client.post(f'/uploads/{jid}/complete',json={},headers=headers()).status_code==409
    assert client.post(url+'0',content=body0,headers=headers()).status_code==200
    assert client.post(url+'0',content=body0,headers=headers()).status_code==200
    assert client.post(url+'0',content=b'c'*20000,headers=headers()).status_code==409
    assert client.get(f'/uploads/{jid}').json()['completed_chunks']==[0,1]
    monkeypatch.setattr(uploads,'probe_video',lambda *a: 600)
    result=client.post(f'/uploads/{jid}/complete',json={},headers=headers())
    assert result.status_code==202,result.text
    assert result.json()['state']=='queued'
    assert uploads.finalize_upload(settings,jid)
    directory=settings.data/'jobs'/jid
    assert (directory/'source.mp4').read_bytes()==body0+body1
    assert json.loads((directory/'source-ready.json').read_text())['duration']==600
    assert not (directory/'upload').exists()
    assert client.post(f'/uploads/{jid}/complete',json={},headers=headers()).json()['id']==jid
    with database(settings) as db: assert usage(db,'owner')==1


def test_ownership_and_length_checks_prevent_writing_other_jobs_or_extra_data(upload_client):
    client,settings=upload_client; job=begin(client,100);jid=job['id']
    assert client.post(f'/uploads/{jid}/chunks/0',content=b'x'*101,headers=headers()).status_code==413
    assert client.post(f'/uploads/{jid}/chunks/0',content=b'x'*99,headers=headers()).status_code==422
    client.cookies.set(settings.cookie,'other')
    assert client.get(f'/uploads/{jid}').status_code==404
    assert client.post(f'/uploads/{jid}/chunks/0',content=b'x'*100,headers=headers()).status_code==404
    assert client.post(f'/uploads/{jid}/complete',json={},headers=headers()).status_code==404
    assert client.post(f'/uploads/{jid}/cancel',json={},headers=headers()).status_code==404


def test_invalid_video_releases_allowance_and_cannot_enter_processing_queue(upload_client,monkeypatch):
    import podclipper_web.uploads as uploads
    client,settings=upload_client;job=begin(client,100);jid=job['id']
    assert client.post(f'/uploads/{jid}/chunks/0',content=b'x'*100,headers=headers()).status_code==200
    def invalid(*a): raise ValueError('Upload a video with an audio track')
    monkeypatch.setattr(uploads,'probe_video',invalid)
    assert client.post(f'/uploads/{jid}/complete',json={},headers=headers()).status_code==202
    assert not uploads.finalize_upload(settings,jid)
    with database(settings) as db:
        assert usage(db,'owner')==0
        assert db.execute('SELECT state FROM jobs WHERE id=?',(jid,)).fetchone()[0]=='failed'
    assert not (settings.data/'jobs'/jid/'upload').exists()
    assert client.post(f'/jobs/{jid}/retry',json={},headers=headers()).status_code==409
    with database(settings) as db: assert usage(db,'owner')==0


def test_cancel_and_abandoned_upload_cleanup_return_allowance(upload_client):
    from podclipper_web.uploads import expire_uploads
    client,settings=upload_client;job=begin(client,100);jid=job['id']
    assert client.post(f'/uploads/{jid}/cancel',json={},headers=headers()).status_code==200
    assert client.get('/jobs/'+jid).json()['state']=='cancelled'
    with database(settings) as db: assert usage(db,'owner')==0
    job=begin(client,100,key='upload-request-0000002');jid=job['id']
    with database(settings) as db:db.execute('UPDATE uploads SET updated_at=? WHERE job_id=?',(time.time()-90000,jid))
    expire_uploads(settings)
    with database(settings) as db:
        assert usage(db,'owner')==0
        assert db.execute('SELECT state FROM jobs WHERE id=?',(jid,)).fetchone()[0]=='expired'


def test_monthly_quota_is_shared_between_uploads_and_youtube(upload_client):
    client,settings=upload_client
    for i in range(5):begin(client,100,key=f'upload-quota-key-{i:03d}')
    assert client.post('/uploads',json={'filename':'six.mp4','size':100,'request_key':'upload-request-six'},headers=headers()).status_code==429
    assert client.post('/jobs',json={'url':'https://youtu.be/abcdefghijk','request_key':'youtube-request-six'},headers=headers()).status_code==429


def test_non_upload_json_requests_keep_existing_small_body_limit(upload_client):
    client,_=upload_client
    assert client.post('/jobs',content=b'x'*20000,headers=headers()).status_code==413


def test_existing_youtube_jobs_survive_schema_migration(upload_client):
    client, settings=upload_client
    job=client.post('/jobs',json={'url':'https://youtu.be/abcdefghijk','request_key':'existing-youtube-job'},headers=headers()).json()
    with database(settings) as db:db.execute('ALTER TABLE jobs DROP COLUMN source_type')
    initialize(settings)
    with database(settings) as db:
        row=db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone()
        assert row['source_type']=='youtube' and row['user_id']=='owner'


def test_corrupt_saved_chunk_is_requested_again_instead_of_queueing_bad_bytes(upload_client,monkeypatch):
    import podclipper_web.uploads as uploads
    client,settings=upload_client;job=begin(client,100);jid=job['id']
    body=b'x'*100
    assert client.post(f'/uploads/{jid}/chunks/0',content=body,headers=headers()).status_code==200
    (settings.data/'jobs'/jid/'upload/chunks/0000.part').write_bytes(b'changed')
    assert client.post(f'/uploads/{jid}/complete',json={},headers=headers()).status_code==202
    assert not uploads.finalize_upload(settings,jid)
    assert client.get(f'/uploads/{jid}').json()['completed_chunks']==[]
    assert client.post(f'/uploads/{jid}/chunks/0',content=body,headers=headers()).status_code==200
    monkeypatch.setattr(uploads,'probe_video',lambda *a: 60)
    assert client.post(f'/uploads/{jid}/complete',json={},headers=headers()).json()['state']=='queued'
    assert uploads.finalize_upload(settings,jid)


def test_completion_only_registers_validation_without_reading_the_video(upload_client, monkeypatch):
    import podclipper_web.uploads as uploads
    client,settings=upload_client; job=begin(client,100);jid=job['id']
    assert client.post(f'/uploads/{jid}/chunks/0',content=b'x'*100,headers=headers()).status_code==200
    def no_probe(*args):
        raise AssertionError('HTTP requests must not probe or assemble videos')
    monkeypatch.setattr(uploads,'probe_video',no_probe)
    response=client.post(f'/uploads/{jid}/complete',json={},headers=headers())
    assert response.status_code==202
    assert response.json()['state']=='queued'
    assert response.json()['stage']=='validating'
    assert not (settings.data/'jobs'/jid/'source.mp4').exists()
    assert client.post(f'/uploads/{jid}/complete',json={},headers=headers()).json()['id']==jid


def test_slow_chunk_disk_flush_does_not_block_other_http_requests(upload_client, monkeypatch):
    import asyncio
    import threading
    import httpx
    import podclipper_web.uploads as uploads
    client,settings=upload_client; jid=begin(client,100)['id']
    started,finished=threading.Event(),threading.Event()
    real_flush=uploads.os.fsync
    def slow_flush(fd):
        started.set();time.sleep(.25);real_flush(fd);finished.set()
    monkeypatch.setattr(uploads.os,'fsync',slow_flush)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app),base_url='https://api.example.com',cookies={settings.cookie:'owner'}) as concurrent:
            task=asyncio.create_task(concurrent.post(f'/uploads/{jid}/chunks/0',content=b'x'*100,headers=headers()))
            while not started.is_set():
                await asyncio.sleep(.005)
            response=await concurrent.get('/health')
            assert response.status_code==200
            assert not finished.is_set(),'disk flush blocked the API event loop'
            assert (await task).status_code==200
    asyncio.run(asyncio.wait_for(exercise(),timeout=3))
