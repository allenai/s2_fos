import requests
from unittest import TestCase


HOST = "host.docker.internal"
PORT = 8080
URL = f"http://{HOST}:{PORT}/invocations"


def get_test_request():
    return dict(
        instances=[
            dict(field1="asdf", field2=2.1),
            dict(field1="fdsa", field2=3.2),
            dict(field1="qwer", field2=-1.0),
        ]
    )


def get_expected_response():
    return dict(
        predictions=[
            dict(output_field="asdf:4.2"),
            dict(output_field="fdsa:6.4"),
            dict(output_field="qwer:-2.0"),
        ]
    )


class TestInvocations(TestCase):
    def test__live_invocation(self):
        request = get_test_request()
        expected_response = get_expected_response()

        resp = requests.post(URL, json=request)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), expected_response)
