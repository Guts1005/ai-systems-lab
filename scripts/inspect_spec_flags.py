from vllm.engine.arg_utils import EngineArgs
import argparse

parser = argparse.ArgumentParser()
parser = EngineArgs.add_cli_args(parser)

for action in parser._actions:
    if any(k in action.dest.lower() for k in ["spec", "draft"]):
        print(f"Flag: {action.option_strings}, Dest: {action.dest}, Default: {action.default}")
        print(f"  Help: {action.help}\n")
