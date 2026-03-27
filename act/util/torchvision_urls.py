import torchvision.datasets as _tvd

# Reliable MNIST mirrors that replace the defunct yann.lecun.com host.
_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",   # PyTorch / Meta S3
    "https://storage.googleapis.com/cvdf-datasets/mnist/",  # Google CVDF
]


def configure_mirror_urls() -> None:
    """Configure torchvision.datasets.MNIST.mirrors to use reliable download URLs.

    The original mirror (yann.lecun.com) is no longer reliably reachable.
    Call this function once, early in any entry point that may trigger an MNIST
    download (either directly via torchvision or transitively through abcrown /
    ERAN data loaders).

    This is idempotent – calling it multiple times has no side effects.
    """
    _tvd.MNIST.mirrors = list(_MNIST_MIRRORS)
