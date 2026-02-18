#!/usr/bin/env python3
"""
Run Long and/or Short Sequence Benchmark for multiple orders sequentially.

Usage:
    python scripts_t5/ella/long/run_all_orders.py --type long
    python scripts_t5/ella/long/run_all_orders.py --type short
    python scripts_t5/ella/long/run_all_orders.py --type both
    python scripts_t5/ella/long/run_all_orders.py --type long --orders 4 5 6 --cl_method ella
    python scripts_t5/ella/long/run_all_orders.py --type short --orders 1 2
    python scripts_t5/ella/long/run_all_orders.py --dry_run
"""

import subprocess
import argparse
import sys
from pathlib import Path

# Default orders per sequence type
DEFAULT_ORDERS_LONG = [5, 6, 4]
DEFAULT_ORDERS_SHORT = [1, 2, 3]


def log_print(*args, **kwargs):
    """Print and flush immediately for nohup compatibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


def run_order_long(order: int, run_name: str, run_number: int, seed: int, cl_method: str,
                   gpu_id: int, dry_run: bool, wandb_project: str = "CL") -> bool:
    """Run a single long order (4, 5, or 6) and return success status."""
    script_path = Path(__file__).parent / "run_long_sequence.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--order", str(order),
        "--run_name", run_name,
        "--run_number", str(run_number),
        "--seed", str(seed),
        "--cl_method", cl_method,
        "--gpu_id", str(gpu_id),
        "--wandb_project", wandb_project,
    ]
    if dry_run:
        cmd.append("--dry_run")
    log_print(f"\n{'#'*70}")
    log_print(f"# Starting Long Order {order}")
    log_print(f"# Command: {' '.join(cmd)}")
    log_print(f"{'#'*70}\n")
    try:
        subprocess.run(cmd, check=True)
        log_print(f"\n✓ Long Order {order} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        log_print(f"\n✗ Long Order {order} failed with return code {e.returncode}\n")
        return False


def run_order_short(order: int, run_name: str, run_number: int, seed: int,
                    gpu_id: int, dry_run: bool, wandb_project: str = "CL") -> bool:
    """Run a single short order (1, 2, or 3) and return success status."""
    script_path = Path(__file__).parent / "run_short_sequence.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--order", str(order),
        "--run_name", run_name,
        "--run_number", str(run_number),
        "--seed", str(seed),
        "--gpu_id", str(gpu_id),
        "--wandb_project", wandb_project,
    ]
    if dry_run:
        cmd.append("--dry_run")
    log_print(f"\n{'#'*70}")
    log_print(f"# Starting Short Order {order}")
    log_print(f"# Command: {' '.join(cmd)}")
    log_print(f"{'#'*70}\n")
    try:
        subprocess.run(cmd, check=True)
        log_print(f"\n✓ Short Order {order} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        log_print(f"\n✗ Short Order {order} failed with return code {e.returncode}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Long and/or Short Sequence Benchmark for multiple orders sequentially"
    )
    parser.add_argument("--type", dest="sequence_type", type=str, default="long",
                        choices=["long", "short", "both"],
                        help="Sequence type: long (orders 4,5,6), short (orders 1,2,3), or both")
    parser.add_argument("--orders", type=int, nargs="+", default=None,
                        help="Orders to run. Default: 4,5,6 for long; 1,2,3 for short; ignored for both")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name (default: long_round1 for long/both, short_round1 for short)")
    parser.add_argument("--run_number", type=int, default=1,
                        help="Run number for output directory naming")
    parser.add_argument("--seed", type=int, default=73,
                        help="Random seed")
    parser.add_argument("--cl_method", type=str, default="ella", choices=["ella", "olora"],
                        help="Continual learning method (long/both only)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--wandb_project", type=str, default="CL",
                        help="Wandb project name (for long/short runs)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--continue_on_failure", action="store_true",
                        help="Continue to next order even if current one fails")

    args = parser.parse_args()

    # Resolve default orders and run_name by type
    if args.sequence_type == "both":
        short_orders = DEFAULT_ORDERS_SHORT
        long_orders = DEFAULT_ORDERS_LONG
        short_run_name = args.run_name or "short_round1"
        long_run_name = args.run_name or "long_round1"
    elif args.sequence_type == "long":
        long_orders = args.orders if args.orders is not None else DEFAULT_ORDERS_LONG
        short_orders = []
        long_run_name = args.run_name or "long_round1"
        short_run_name = None
    else:  # short
        short_orders = args.orders if args.orders is not None else DEFAULT_ORDERS_SHORT
        long_orders = []
        short_run_name = args.run_name or "short_round1"
        long_run_name = None

    log_print(f"\n{'='*70}")
    log_print(f"Sequence type: {args.sequence_type}")
    if short_orders:
        log_print(f"Short orders: {short_orders}")
    if long_orders:
        log_print(f"Long orders: {long_orders}")
    if long_orders:
        log_print(f"CL Method: {args.cl_method}")
    log_print(f"Run Number: {args.run_number}")
    log_print(f"Seed: {args.seed}")
    log_print(f"{'='*70}\n")

    results = {}  # key: "short_1", "long_5", etc.

    def run_orders(orders, run_name, is_short):
        for order in orders:
            key = f"{'short' if is_short else 'long'}_{order}"
            if is_short:
                success = run_order_short(
                    order=order,
                    run_name=run_name,
                    run_number=args.run_number,
                    seed=args.seed,
                    gpu_id=args.gpu_id,
                    dry_run=args.dry_run,
                    wandb_project=args.wandb_project,
                )
            else:
                success = run_order_long(
                    order=order,
                    run_name=run_name,
                    run_number=args.run_number,
                    seed=args.seed,
                    cl_method=args.cl_method,
                    gpu_id=args.gpu_id,
                    dry_run=args.dry_run,
                    wandb_project=args.wandb_project,
                )
            results[key] = success
            if not success and not args.continue_on_failure:
                return False
        return True

    if args.sequence_type == "both":
        if not run_orders(short_orders, short_run_name, is_short=True):
            log_print("Stopping due to failure in short orders")
        elif not run_orders(long_orders, long_run_name, is_short=False):
            log_print("Stopping due to failure in long orders")
    elif short_orders:
        run_orders(short_orders, short_run_name, is_short=True)
    else:
        run_orders(long_orders, long_run_name, is_short=False)

    # Print summary
    log_print(f"\n{'='*70}")
    log_print("Summary")
    log_print(f"{'='*70}")
    for key, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        log_print(f"  {key}: {status}")
    log_print(f"{'='*70}\n")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
