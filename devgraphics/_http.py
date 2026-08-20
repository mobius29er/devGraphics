"""
The small amount of HTTP every backend needs, written once.

Six backends make JSON requests over urllib. Without this module each of them
grows its own copy of "decode the error body, work out whether a retry is
sensible, back off". They would drift, and the ones that drift are the paid ones.

Two things here are not obvious and are worth stating.

`Retry-After` is honoured but capped. OpenAI's image tiers are rated in images per
minute, so a 429 during an 88-icon batch is normal operation rather than a fault;
sleeping through it is correct. A server that answers `Retry-After: 3600` is not,
so the cap keeps a stuck run interruptible.

Errors are classified, not just raised. `moderation_blocked` is non-retryable --
retrying a refused prompt burns money to get the same refusal -- and a 402 means
the account is out of credit, which is fatal for the whole batch rather than for
one icon. Both need to be distinguishable from a transient 500 at the call site.

Pure stdlib. devgraphics ships three dependencies and none of them is an HTTP
client; adding `requests` so that six short functions read slightly nicer is a
bad trade.
"""

import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid

from .backends.base import (AuthError, BackendError, ModerationBlocked,
                            PaymentRequired, RateLimited)

USER_AGENT = "devgraphics"

#: Cap on an honoured Retry-After, in seconds. Longer than the longest rate-limit
#: window anybody publishes, short enough that Ctrl-C still feels responsive.
MAX_BACKOFF = 120.0


def request_json(url, payload=None, headers=None, method=None, timeout=300,
                 retries=4, sleep=time.sleep):
    """One JSON round trip, with backoff on the failures worth retrying.

    `payload` None means GET (or whatever `method` says) with no body.
    Returns the decoded JSON. Raises a BackendError subclass on failure.
    """
    body = None
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    raw = _retrying(url, body, hdrs, method, timeout, retries, sleep)
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise BackendError("%s returned a non-JSON body: %s"
                           % (url, raw[:200])) from exc


def request_bytes(url, payload=None, headers=None, method=None, timeout=300,
                  retries=4, sleep=time.sleep):
    """As `request_json`, but the response is raw bytes -- an image, usually."""
    body = None
    hdrs = {"User-Agent": USER_AGENT}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    return _retrying(url, body, hdrs, method, timeout, retries, sleep)


def post_multipart(url, fields, files, headers=None, timeout=300, retries=2,
                   sleep=time.sleep):
    """multipart/form-data, because ComfyUI's /upload/image takes nothing else.

    `fields` is {name: str}, `files` is {name: (filename, bytes)}. Hand-rolled
    because the stdlib has no encoder and pulling in requests-toolbelt for thirty
    lines would be silly.
    """
    boundary = "----devgraphics%s" % uuid.uuid4().hex
    out = []
    for name, value in (fields or {}).items():
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                    % (boundary, name, value)).encode("utf-8"))
    for name, (filename, data) in (files or {}).items():
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; "
                    "filename=\"%s\"\r\nContent-Type: %s\r\n\r\n"
                    % (boundary, name, filename, ctype)).encode("utf-8"))
        out.append(data)
        out.append(b"\r\n")
    out.append(("--%s--\r\n" % boundary).encode("utf-8"))
    body = b"".join(out)

    hdrs = {"User-Agent": USER_AGENT,
            "Content-Type": "multipart/form-data; boundary=%s" % boundary}
    hdrs.update(headers or {})
    raw = _retrying(url, body, hdrs, "POST", timeout, retries, sleep)
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        return {}


def _retrying(url, body, headers, method, timeout, retries, sleep):
    delay = 1.0
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers,
                                         method=method or ("POST" if body else "GET"))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            detail = _detail(exc)
            err = _classify(exc.code, detail, exc.headers)
            if not _retryable(err) or attempt == retries:
                raise err
            last = err
            wait = getattr(err, "retry_after", None) or delay
            sleep(min(float(wait), MAX_BACKOFF))
            delay = min(delay * 2, MAX_BACKOFF)
        except urllib.error.URLError as exc:
            last = BackendError("cannot reach %s: %s" % (url, exc.reason))
            if attempt == retries:
                raise last
            sleep(min(delay, MAX_BACKOFF))
            delay = min(delay * 2, MAX_BACKOFF)
    raise last                                            # pragma: no cover


def _detail(exc):
    """The provider's own error text, which is always more useful than the code."""
    try:
        raw = exc.read()
    except Exception:                                     # pragma: no cover
        return ""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError:
        return raw.decode("utf-8", "replace")[:400]
    err = doc.get("error", doc)
    if isinstance(err, dict):
        return "%s%s" % (err.get("message") or err.get("detail") or json.dumps(doc)[:300],
                         " [%s]" % err["code"] if err.get("code") else "")
    if isinstance(err, list):                             # FastAPI validation
        return json.dumps(err)[:400]
    return str(err)[:400]


def _classify(code, detail, headers=None):
    text = "HTTP %d: %s" % (code, detail)
    if "moderation_blocked" in detail or "content_policy" in detail:
        return ModerationBlocked(text + "\n  non-retryable; change the prompt")
    if code in (401, 403):
        return AuthError(text)
    if code == 402:
        return PaymentRequired(text + "\n  fatal for the batch, not just this icon")
    if code == 429:
        after = None
        if headers is not None:
            try:
                after = float(headers.get("Retry-After") or 0) or None
            except (TypeError, ValueError):
                after = None
        return RateLimited(text, retry_after=after)
    return BackendError(text)


def _retryable(err):
    if isinstance(err, RateLimited):
        return True
    if isinstance(err, (AuthError, PaymentRequired, ModerationBlocked)):
        return False
    message = str(err)
    return any(("HTTP %d" % c) in message for c in (408, 500, 502, 503, 504, 520, 524))
