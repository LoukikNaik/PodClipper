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
            db.execute("INSERT INTO reels VALUES ('reel','other-job','Test reel','reels/reel.mp4',10,?)", (time.time(),))
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
        self.login('owner-session')
        self.assertTrue(self.client.get('/me').json()['is_admin'])
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


if __name__ == '__main__':
    unittest.main()
