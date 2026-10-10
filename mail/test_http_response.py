"""Loopback-only response framing check with a large synthetic export."""
from functools import partial
from http.server import ThreadingHTTPServer
import threading
import unittest
from unittest.mock import Mock

import requests

from dashboard import Handler


class HTTPResponseCheck(unittest.TestCase):
    def test_large_export_preserves_full_body_and_content_length(self):
        payload = b'{"state":"synthetic export"}\n' * 220000 + b'\n'
        app = Mock()
        app.authorized.return_value = True
        app.training_export.return_value = payload
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        origin = "http://127.0.0.1:" + str(server.server_port)
        server.RequestHandlerClass = partial(Handler, app=app, origin=origin)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with requests.Session() as client:
                client.trust_env = False
                response = client.get(origin + "/api/training-export", timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["Content-Length"], str(len(payload)))
            self.assertEqual(response.headers["Content-Type"], "application/x-ndjson; charset=utf-8")
            self.assertEqual(response.content, payload)
            app.training_export.assert_called_once_with(split=None)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
