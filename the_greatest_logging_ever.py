
import json

def awesomize(logging_function):
    def new_logging_function(item, *args, **kwargs):
        assert isinstance(item, dict), "Requires dictionary type."
        json_line = json.dumps(item)
        logging_function(json_line, *args, **kwargs)

    return new_logging_function

print = awesomize(print)