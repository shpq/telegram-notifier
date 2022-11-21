import traceback


def get_error_message(error) -> str:
    """
    Returns error message for error with simple structure
    """
    message = str(error.__class__.__name__)
    traceback_message = traceback.format_exc()
    if error.args:
        message += f": {error.args[0]}"
    message += f"\n{traceback_message}"
    return message
