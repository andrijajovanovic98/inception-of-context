import warnings

# Starlette 1.x announces that its TestClient will move from httpx to httpx2.
# The warning is about the test tooling FastAPI ships, not about IoC.
warnings.filterwarnings("ignore", message=r"Using `httpx` with `starlette\.testclient` is deprecated")
