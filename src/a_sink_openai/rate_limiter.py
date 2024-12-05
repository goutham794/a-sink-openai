import asyncio
import re
import time

from loguru import logger


# Define a regular expression to capture time components
REGEX_TIME = re.compile(r"(?P<value>\d+)(?P<unit>[smhms]+)")

RATE_LIMITER = None
MAX_SLEEP_TIME = 2 * 60  # 2 Minutes


def set_rate_limiter(max_requests: int = None, max_tokens: int = None):
    global RATE_LIMITER
    RATE_LIMITER = RateLimiter(max_requests, max_tokens)


class RateLimiter:
    def __init__(self, max_requests: int = None, max_tokens: int = None):
        self.max_requests = max_requests
        self.max_tokens = max_tokens

        self.remaining_requests = max_requests
        self.remaining_tokens = max_tokens

        # Assume reset in 60 seconds initially
        self.reset_time_requests = time.monotonic() + 60
        self.reset_time_tokens = time.monotonic() + 60

        logger.info(f"Setting {self}")

    def __repr__(self):
        max_reqs = self.max_requests
        max_tokens = self.max_tokens
        rem_reqs = self.remaining_requests
        rem_tokens = self.remaining_tokens
        reset_t_reqs = round(self.reset_time_requests)
        reset_t_tokens = round(self.reset_time_tokens)

        if max_tokens is None:
            return f"RateLimiter({max_reqs=}, {rem_reqs=}, {reset_t_reqs=} [no_tokens])"

        return (
            f"RateLimiter({max_reqs=}, {max_tokens=} "
            f"{rem_reqs=}, {rem_tokens=}, {reset_t_reqs=}, {reset_t_tokens=})"
        )

    def _get_seconds_to_sleep(self):
        if self.max_tokens is None:
            # Only consider requests if tokens are not being used
            sleep_time = self.reset_time_requests - time.monotonic()
        else:
            sleep_time = min(
                self.reset_time_requests - time.monotonic(),
                self.reset_time_tokens - time.monotonic(),
            )

        # Do not wait really high times. Better to try anyway
        sleep_time = min(sleep_time, MAX_SLEEP_TIME)

        # Ensure sleep_time is at least 1 second
        return max(sleep_time, 1)

    async def wait_for_availability(self, required_tokens=None):
        if self.max_tokens is None:
            required_tokens = 0  # Ignore tokens if max_tokens is None

        while self.remaining_requests <= 0 or (
            self.max_tokens is not None and self.remaining_tokens < required_tokens
        ):
            self.update_limits()

            seconds_to_sleep = self._get_seconds_to_sleep()
            logger.debug(f"Sleeping {seconds_to_sleep=}")
            await asyncio.sleep(seconds_to_sleep)

        self.remaining_requests -= 1
        if self.max_tokens is not None:
            self.remaining_tokens -= required_tokens

    def update_limits(self):
        current_time = time.monotonic()

        # If we've passed the reset time, reset the limits
        if current_time >= self.reset_time_requests:
            self.remaining_requests = self.max_requests
            self.reset_time_requests = current_time + 60  # Reset the time window

        if self.max_tokens is not None and current_time >= self.reset_time_tokens:
            self.remaining_tokens = self.max_tokens
            self.reset_time_tokens = current_time + 60  # Reset the time window

        logger.debug(f"Updating limits:\n{self}")

    def _parse_reset_time(self, reset_time_str):
        """Convert a time reset string like '1s', '6m0s', or '60ms' into seconds."""

        total_seconds = 0
        for match in REGEX_TIME.finditer(reset_time_str):
            value = int(match.group("value"))
            unit = match.group("unit")

            if unit == "s":
                total_seconds += value
            elif unit == "m":
                total_seconds += value * 60
            elif unit == "h":
                total_seconds += value * 3600
            elif unit == "ms":
                total_seconds += value / 1000.0

        # Default to 60 seconds if the format is unexpected
        return total_seconds if total_seconds > 0 else 60

    def update_from_headers(self, headers):
        """Update the rate limits based on headers from the API response."""

        self.remaining_requests = int(
            headers.get("x-ratelimit-remaining-requests", self.remaining_requests)
        )

        if self.max_tokens is not None:
            self.remaining_tokens = int(
                headers.get("x-ratelimit-remaining-tokens", self.remaining_tokens)
            )

            reset_tokens_seconds = self._parse_reset_time(
                headers.get("x-ratelimit-reset-tokens", "60s")
            )
            self.reset_time_tokens = time.monotonic() + reset_tokens_seconds

        reset_requests_seconds = self._parse_reset_time(
            headers.get("x-ratelimit-reset-requests", "60s")
        )

        self.reset_time_requests = time.monotonic() + reset_requests_seconds
        logger.debug(f"Updating limits from headers:\n{self}")