import time
import server
import execution
from  ..logger import logger, AnsiColor
from ..config import config

_original_promptExecutor_execute = execution.PromptExecutor.execute
_original_execute = execution.execute

_total_execution_time = 0
_execution_times = {}

def send_event(data):
    try:
        server.PromptServer.instance.send_sync(
            "NNProfiler",
            data
        )
    except Exception:
        pass

def get_config(key):
    return getattr(config.profiler, key)

def clean_id(node_id):
    return str(node_id).lstrip("#")

def parent_chain(node_id):
    """
    "11:10:9" -> ["11:10", "11"]
    "11:10"   -> ["11"]
    "11"      -> []
    """
    result = []
    current = clean_id(node_id)

    while ":" in current:
        current = current.rsplit(":", 1)[0]
        result.append(current)

    return result

def reset():
    nodes = set(_execution_times.keys())

    send_event({
        "type": "node_reset",
        "nodes": list(nodes)
    })
   
    _execution_times.clear()


def get_class_type(args, node_id):

    try:
        prompt = args[1]
        node = prompt.original_prompt.get(
            str(node_id),
            {}
        )

        return node.get(
            "class_type",
            "unknown"
        )

    except Exception:
        return "unknown"


async def profiled_execute(*args, **kwargs):

    enable = get_config("enable")
    if not enable:
        return await _original_execute(
            *args,
            **kwargs
        )

    node_id = clean_id(args[3])
    class_type = get_class_type(
        args,
        node_id
    )
   
    # START
    start = time.perf_counter()

    send_event({
        "type": "node_start",
        "node": node_id
    })

    # parent subgraphs
    parents = parent_chain(node_id)

    for parent in parents:
        send_event({
            "type": "subgraph_start",
            "node": parent,
            "child": node_id,
            "total": _execution_times.get(
                parent,
                0
            ),
            "class_type": class_type
        })

    try:
        result = await _original_execute(
            *args,
            **kwargs
        )

        return result


    finally:
        end = time.perf_counter()
        elapsed = end - start        

        # NODE FINISH

        _execution_times[node_id] = elapsed

        global _total_execution_time
        _total_execution_time = _total_execution_time + elapsed

        send_event({
            "type": "node_time",
            "node": node_id,
            "time": elapsed,
            "class_type": class_type
        })

        # PROPAGATE TO PARENTS

        for parent in parents:
            _execution_times[parent] = (
                _execution_times.get(parent, 0)
                + elapsed
            )

            send_event({
                "type": "subgraph_time",
                "node": parent,
                "child": node_id,
                "total": _execution_times[parent]
            })

        # CONSOLE

        console_log = get_config("console_log")
        threshold = get_config("console_log_threshold")  

        if console_log and elapsed >= float(threshold):
            precision = get_config("precision")
            slow_time = get_config("slow_time")

            if elapsed < slow_time / 2:
                color = AnsiColor.GREEN
            elif elapsed < slow_time:
                color = AnsiColor.YELLOW
            else:
                color = AnsiColor.RED

            logger.info(
                f"{AnsiColor.BLUE}[Profiler]{AnsiColor.RESET} "
                f"{AnsiColor.GRAY}#{node_id}{AnsiColor.RESET} "
                f"{class_type}: "
                f"{color}{elapsed:.{precision}f}s{AnsiColor.RESET}"
            )

def promptExecutor_execute(self, *args, **kwargs):
    # Before execute
    reset()
    config.update()

    global _total_execution_time
    _total_execution_time = 0

    enable = get_config("enable") 
    console_log = get_config("console_log")

    if enable:
        send_event({
            "type": "config",
            "slowTime": get_config('slow_time'),
            "activeColor": get_config('active_color'),
            "badgeColor": get_config('badge_color'),
            "badgeColorSlow": get_config('badge_color_slow'),       
            "precision": get_config('precision'),       
        })

    result =  _original_promptExecutor_execute(self, *args, **kwargs)

    # After execute

    if enable and console_log:
        precision = get_config("precision") 
        logger.info(
            f"{AnsiColor.BLUE}[Profiler]{AnsiColor.RESET} "       
            f"Nodes executed in "
            f"{AnsiColor.GREEN}{_total_execution_time:.{precision}f}s{AnsiColor.RESET}"
        )

    return result


def install_hook():

    if getattr(execution.execute, "_simple_profiler", False):
        return
    
    execution.PromptExecutor.execute = promptExecutor_execute
    execution.execute = profiled_execute
    execution.execute._simple_profiler = True    

    print(
        "[SimpleProfiler] execution hook installed"
    )