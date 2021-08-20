import subprocess
import requests
from unittest import TestCase


HOST = str(
    subprocess.run(
        "/sbin/ip route|awk '/default/ { print $3 }'", shell=True, capture_output=True
    ).stdout.strip(),
    "ascii",
)
PORT = 8080
URL = f"http://{HOST}:{PORT}/invocations"


class TestInvocations(TestCase):
    def test__live_invocation(self):
        pass
