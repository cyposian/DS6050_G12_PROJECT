import argparse
import sys
import os
import socket

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="G12 Fashion-MNIST Training Harness")

    # Required argument
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    # Optional argument for the message
    parser.add_argument("--message", type=str, default="Hello from Slurm Python job!",
                        help="Message to print")

    args = parser.parse_args()

    #if len(sys.argv) > 1:
    #    message = sys.argv[1]
    #else:
    #    message = "Hello from Slurm Python job!"
        
    print(f"Job started on host: {socket.gethostname()}")
    print(f"Selected Config/YAML File: {args.config}")
    print(f"Message: {args.message}")
    print("Job completed successfully.")

