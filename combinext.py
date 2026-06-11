import os
import glob
import argparse

def combine_extxyz(input_dir, output_file="combined.extxyz"):
    """Find all .extxyz files in a directory and combine them into one file."""
    # Find all .extxyz files
    pattern = os.path.join(input_dir, "**", "*.extxyz")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"No .extxyz files found in {input_dir}")
        return

    print(f"Found {len(files)} .extxyz files")

    total_frames = 0
    with open(output_file, "w") as outfile:
        for filepath in files:
            print(f"  Adding: {filepath}")
            with open(filepath, "r") as infile:
                content = infile.read()
                outfile.write(content)
                # Ensure there's a newline between files
                if not content.endswith("\n"):
                    outfile.write("\n")

            # Count frames (each frame starts with an atom count line)
            with open(filepath, "r") as infile:
                while True:
                    line = infile.readline()
                    if not line:
                        break
                    try:
                        n_atoms = int(line.strip())
                        total_frames += 1
                        # Skip comment line + atom lines
                        infile.readline()  # comment
                        for _ in range(n_atoms):
                            infile.readline()
                    except ValueError:
                        continue

    print(f"Combined {total_frames} frames into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine .extxyz files")
    parser.add_argument("input_dir", help="Directory to search for .extxyz files")
    parser.add_argument("-o", "--output", default="combined.extxyz", help="Output file (default: combined.extxyz)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories")
    args = parser.parse_args()

    if args.no_recursive:
        # Override to non-recursive
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.extxyz")))
        with open(args.output, "w") as outfile:
            for f in files:
                with open(f) as infile:
                    content = infile.read()
                    outfile.write(content)
                    if not content.endswith("\n"):
                        outfile.write("\n")
        print(f"Combined {len(files)} files into {args.output}")
    else:
        combine_extxyz(args.input_dir, args.output)