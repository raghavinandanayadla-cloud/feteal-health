"""Lab 7: package the selected model for local API inference."""

from src.prepare_deployment import prepare_deployment


def run() -> dict:
    metadata = prepare_deployment()
    print(f"Packaged {metadata['model_file']} ({metadata['sha256']})")
    return metadata


if __name__ == "__main__":
    run()