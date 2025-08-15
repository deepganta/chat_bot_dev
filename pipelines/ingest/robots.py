from urllib.parse import urlparse
from urllib import robotparser

class RobotsGate:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._parsers = {}

    def _origin(self, url: str) -> str:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    def _get_parser(self, url: str) -> robotparser.RobotFileParser:
        origin = self._origin(url)
        if origin not in self._parsers:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable, fail open by default
                rp = None
            self._parsers[origin] = rp
        return self._parsers[origin]

    def allowed(self, url: str) -> bool:
        rp = self._get_parser(url)
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True
