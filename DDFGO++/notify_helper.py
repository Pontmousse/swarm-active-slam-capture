try:
    import requests
except Exception:  # optional dependency
    requests = None


def notify(message, title="DDFGO++ Update", topic="DDFGO-standalone", verbose=True):
    """
    Send a best-effort notification via ntfy.sh.
    Non-fatal: if requests/network is unavailable, silently skip.
    """
    if not verbose or requests is None:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=str(message).encode("utf-8"),
            headers={"Title": title},
            timeout=10,
        )
    except Exception:
        # Keep runtime non-fatal by design.
        pass
