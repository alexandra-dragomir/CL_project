import time
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run a command after a delay in hours.")
    parser.add_argument("hours", type=float, help="Delay in hours before running the command")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command to run after the delay (e.g. python3 script.py arg1 arg2)")
    args = parser.parse_args()

    delay_seconds = int(args.hours * 3600)
    # delay_seconds = 10
    print(f"Sleeping for {delay_seconds} seconds (~{args.hours} hours)...")
    print(f"Command to run: {' '.join(args.command)}")
    time.sleep(delay_seconds)

    print(f"Launching command: {' '.join(args.command)}")
    subprocess.Popen(args.command)

if __name__ == "__main__":
    main()