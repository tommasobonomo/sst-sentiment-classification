import logging

# Define logger object
console_logger = logging.getLogger()
console_logger.setLevel(logging.INFO)

# Define console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s: %(message)s"))
console_logger.addHandler(console_handler)
