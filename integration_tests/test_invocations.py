import requests
from unittest import TestCase


HOST = "host.docker.internal"
PORT = 8080
URL = f"http://{HOST}:{PORT}/invocations"


class TestInvocations(TestCase):
    def test__live_invocation(self):
        raise NotImplementedError()
