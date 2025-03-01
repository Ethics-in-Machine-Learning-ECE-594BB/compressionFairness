import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hello", type = str)
args = parser.parse_args()
print(f"hello {args.hello}")