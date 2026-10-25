"""Admin access is server-side and does not weaken ordinary job ownership."""

import hashlib
from importlib.util import find_spec
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

_WEB_TESTS_AVAILABLE = find_spec('fastapi') is not None and find_spec('httpx') is not None
if _WEB_TESTS_AVAILABLE:
    from fastapi.testclient import TestClient
    from podclipper_web.api import create_app
from podclipper_web.store import Settings, database, initialize


@unittest.skipUnless(_WEB_TESTS_AVAILABLE, 'Install the web extra and httpx to run web API checks')
class AdminDashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = Settings(Path(self.directory.name).resolve(), 'test.apps.googleusercontent.com',
                                 'https://podclipper.loukik.dev', True, 10800, 7)
        self.settings.admin_user_id = 'owner-sub'
        initialize(self.settings)
        with database(self.settings) as db:
            for user_id, email, token in [('owner-sub', 'owner@example.com', 'owner-session'),
                                          ('another-sub', 'owner@example.com', 'other-session')]:
                db.execute('INSERT INTO users VALUES (?,?,?,?)', (user_id, email, user_id, time.time()))
                db.execute('INSERT INTO sessions VALUES (?,?,?,?)',
                           (hashlib.sha256(token.encode()).hexdigest(), user_id, 'csrf', time.time() + 60))
            db.execute("INSERT INTO jobs (id,user_id,video_id,url,clips,quota_month,request_key,created_at,updated_at) "
                       "VALUES ('other-job','another-sub','abcdefghijk','https://youtu.be/abcdefghijk',5,'2026-10','request',?,?)",
                       (time.time(), time.time()))
            db.execute("INSERT INTO reels (id,job_id,title,path,duration,created_at) VALUES ('reel','other-job','Test reel','reels/reel.mp4',10,?)", (time.time(),))
        output = self.settings.data / 'jobs' / 'other-job' / 'reels'
        output.mkdir(parents=True)
        (output / 'reel.mp4').write_bytes(b'test-video-data')
        (output / 'reel.poster.jpg').write_bytes(b'\xff\xd8test-poster\xff\xd9')
        self.patch = patch.object(Settings, 'load', return_value=self.settings)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = self.enterContext(TestClient(create_app(), base_url='https://api.example.com'))

    def login(self, token):
        self.client.cookies.set(self.settings.cookie, token)

    def test_only_configured_google_subject_can_view_admin_jobs(self):
        self.assertEqual(self.client.get('/admin/jobs').status_code, 401)
        self.login('other-session')
        self.assertEqual(self.client.get('/admin/jobs').status_code, 403)
        self.assertFalse(self.client.get('/me').json()['is_admin'])
        self.assertEqual(self.client.get('/me').json()['retention_days'], 7)
        self.login('owner-session')
        self.assertTrue(self.client.get('/me').json()['is_admin'])
        self.assertIsNone(self.client.get('/me').json()['retention_days'])
        response = self.client.get('/admin/jobs')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['total'], 1)
        self.assertEqual(response.json()['jobs'][0]['owner']['email'], 'owner@example.com')
        self.assertEqual(self.client.get('/admin/jobs?offset=50').json()['jobs'], [])

    def test_admin_media_uses_its_own_guard_and_preserves_normal_ownership(self):
        self.login('other-session')
        self.assertEqual(self.client.get('/admin/jobs/other-job').status_code, 403)
        for method in (self.client.get, self.client.head):
            self.assertEqual(method('/admin/jobs/other-job/reels/reel').status_code, 403)
        self.login('owner-session')
        self.assertEqual(self.client.get('/jobs/other-job').status_code, 404)
        self.assertEqual(self.client.get('/jobs/other-job/reels/reel').status_code, 404)
        self.assertEqual(self.client.get('/admin/jobs/other-job').status_code, 200)
        result = self.client.get('/admin/jobs/other-job/reels/reel', headers={'Range': 'bytes=0-3'})
        self.assertEqual(result.status_code, 206)
        self.assertEqual(result.content, b'test')
        self.assertEqual(self.client.head('/admin/jobs/other-job/reels/reel').status_code, 200)
        self.assertEqual(self.client.get('/admin/jobs/missing').status_code, 404)

    def test_no_admin_is_allowed_when_server_configuration_is_empty(self):
        self.settings.admin_user_id = ''
        self.login('owner-session')
        self.assertEqual(self.client.get('/admin/jobs').status_code, 403)

    def test_poster_is_visible_before_playback_but_keeps_media_permissions(self):
        url = '/jobs/other-job/reels/reel?poster=true'
        self.assertEqual(self.client.get(url).status_code, 401)
        self.login('other-session')
        result = self.client.get(url)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers['content-type'], 'image/jpeg')
        self.assertEqual(result.content, b'\xff\xd8test-poster\xff\xd9')
        self.assertEqual(result.headers['cache-control'], 'private, no-store')
        self.login('owner-session')
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(self.client.get('/admin/jobs/other-job/reels/reel?poster=true').headers['content-type'], 'image/jpeg')
        self.assertEqual(self.client.head('/admin/jobs/other-job/reels/reel?poster=true').status_code, 200)
        self.login('other-session')
        self.assertEqual(self.client.get('/admin/jobs/other-job/reels/reel?poster=true').status_code, 403)

    def test_submission_without_reel_count_uses_automatic_selection(self):
        self.login('other-session')
        payload = {'url': 'https://youtu.be/abcdefghij2', 'request_key': 'automatic-request-0001'}
        headers = {'Origin': self.settings.origin, 'X-CSRF-Token': 'csrf'}
        result = self.client.post('/jobs', json=payload, headers=headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['clips'], 0)
        self.assertIsNone(result.json()['selected_count'])
        self.assertEqual(self.client.get('/config').json()['max_reels'], 30)
        again = self.client.post('/jobs', json=payload, headers=headers)
        self.assertEqual(again.json()['id'], result.json()['id'])

    def test_partial_selection_exposes_approved_count_without_claiming_discovery_is_finished(self):
        import json
        directory = self.settings.data / 'jobs' / 'other-job'
        (directory / 'selection.json').write_text(json.dumps({'clips': [{'start': 20, 'end': 40}],
                                                            'discovery_complete': False}))
        from podclipper_web.worker import sync_metadata
        sync_metadata(self.settings,'other-job')
        self.login('other-session')
        result = self.client.get('/jobs/other-job').json()
        self.assertEqual(result['selected_count'], 1)
        self.assertIs(result['discovery_complete'], False)
        self.assertEqual(len(result['reels']), 1)

    def test_worker_derives_and_saves_cap_after_video_duration_is_known(self):
        import json
        from podclipper_web.worker import prepare_request
        directory = self.settings.data / 'jobs' / 'other-job'
        (directory / 'source-ready.json').write_text(json.dumps({'bytes': 100, 'duration': 7200}))
        job = {'id': 'other-job', 'clips': 0}
        prepare_request(self.settings, job, directory)
        self.assertEqual(job['clips'], 24)
        self.assertEqual(json.loads((directory / 'request.json').read_text())['clips'], 24)
        with database(self.settings) as db:
            self.assertEqual(db.execute("SELECT clips FROM jobs WHERE id='other-job'").fetchone()[0], 24)

    def test_status_responses_do_not_read_progress_or_quality_files(self):
        self.login('other-session')
        with patch.object(Path, 'read_text', side_effect=AssertionError('API must use persisted metadata')):
            result=self.client.get('/jobs/other-job')
        self.assertEqual(result.status_code,200)
        self.assertEqual(len(result.json()['reels']),1)


    def test_admin_listing_batches_database_reads_as_job_count_grows(self):
        from contextlib import contextmanager
        import podclipper_web.api as api
        with database(self.settings) as db:
            for i in range(15):
                db.execute("INSERT INTO jobs (id,user_id,video_id,url,clips,quota_month,request_key,created_at,updated_at) VALUES (?,'another-sub','abcdefghijk','https://youtu.be/abcdefghijk',0,'2026-10',?,0,0)",(f'job-{i}',f'request-{i}'))
        reads=[]
        @contextmanager
        def traced(settings):
            with database(settings) as db:
                db.set_trace_callback(lambda sql:reads.append(sql) if sql.lstrip().upper().startswith('SELECT') else None)
                yield db
        self.login('owner-session')
        with patch.object(api,'database',traced):
            result=self.client.get('/admin/jobs?summary=true')
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.json()['total'],16)
        self.assertLessEqual(len(reads),8)



if __name__ == '__main__':
    unittest.main()
