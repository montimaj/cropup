"""``python -m cropup.web`` -- serve the app with the settings from the environment.

``bootstrap`` is imported before anything else, as SPEC section 2 requires, and
is what sets ``EE_PROJECT_ID``. Host, port and log level come from
``CROPUP_HOST`` / ``CROPUP_PORT`` / ``CROPUP_LOG_LEVEL`` so that one setting
configures uvicorn and ``bootstrap.configure_logging`` alike.

This is also where SPEC 7 stops being about cropup's own logger and starts being
about the process: uvicorn writes an access line per request with the request
target in it, and on this app's own pages that target carries the farmer's typed
autocomplete prefix and the field's exact coordinates. So the redaction filter
goes onto uvicorn's loggers here, before ``uvicorn.run`` can log anything.
"""

from __future__ import annotations

from .. import bootstrap  # imported first on purpose (SPEC 2)
from ..config import get_settings


def main() -> None:
    import uvicorn  # noqa: PLC0415 - a CLI dependency, not an import-time one

    settings = get_settings()
    # Before the server exists, so that the very first request it logs is
    # already redacted. The app's lifespan calls bootstrap.configure_logging(),
    # which installs the same filters, but that runs after uvicorn has
    # configured its own logging -- and "covered from the second request
    # onwards" is not what SPEC 7 says. The filters sit on the loggers, so
    # uvicorn's dictConfig replacing their handlers does not remove them.
    bootstrap.install_privacy_filters()
    uvicorn.run(
        "cropup.web.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
