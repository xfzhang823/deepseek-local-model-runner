from pathlib import Path


def find_project_root(starting_path=None, marker=".git"):
    """
    Recursively find the root directory of the project by looking for a specific marker.
    """
    if starting_path is None:
        starting_path = Path(__file__).resolve().parent

    starting_path = Path(starting_path)

    print(f"\n🔍 Searching for project root starting from: {starting_path}")

    # ✅ First, check if the marker exists in the current directory
    marker_path = starting_path / marker
    print(f"🟡 Checking: {marker_path} → Exists? {marker_path.exists()}")

    if marker_path.exists():
        print(f"✅ Found marker in starting directory: {starting_path}")
        return starting_path

    # ✅ Then, check parent directories
    for parent in starting_path.parents:
        marker_path = parent / marker
        print(f"🟡 Checking: {marker_path} → Exists? {marker_path.exists()}")

        if marker_path.exists():
            print(f"✅ Found project root at: {parent}")
            return parent

    print("❌ Project root not found!")
    return None


if __name__ == "__main__":
    print(find_project_root())
