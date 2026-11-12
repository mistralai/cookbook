# Python 2 HTTP client: urllib2, urlparse, httplib.
# Demonstrates the urllib/urlparse split that was unified in Python 3.

import urllib2
import httplib
from urlparse import urlparse, urljoin


class ApiClient(object):

    def __init__(self, base_url, timeout=30):
        self.base_url = base_url
        self.timeout = timeout

    def get(self, path, params=None):
        url = urljoin(self.base_url, path)
        if params:
            query = "&".join("%s=%s" % (k, v) for k, v in params.iteritems())
            url = url + "?" + query
        try:
            response = urllib2.urlopen(url, timeout=self.timeout)
            return response.read()
        except urllib2.HTTPError as e:
            raise RuntimeError("HTTP %d: %s" % (e.code, e.reason))
        except urllib2.URLError as e:
            raise RuntimeError("Request failed: %s" % e.reason)

    def post(self, path, data):
        url = urljoin(self.base_url, path)
        parsed = urlparse(url)
        conn = httplib.HTTPConnection(parsed.netloc, timeout=self.timeout)
        conn.request("POST", parsed.path, data)
        resp = conn.getresponse()
        if resp.status >= 400:
            raise RuntimeError("POST failed with status %d" % resp.status)
        return resp.read()


if __name__ == "__main__":
    client = ApiClient("http://example.com/api/")
    print client.get("users")
