from transactionguard.config import config
from transactionguard.data.synthetic import write_benchmark
from transactionguard.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    frame = write_benchmark(config.RAW_DATA_PATH)
    logger.info(
        "Wrote %s transactions (%s fraud, %.3f%%) to %s",
        len(frame), int(frame["Class"].sum()), 100 * frame["Class"].mean(), config.RAW_DATA_PATH,
    )


if __name__ == "__main__":
    main()
