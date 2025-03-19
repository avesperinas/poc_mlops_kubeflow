import os

from .authentication import DeployKFCredentialsOutOfBand


credentials = DeployKFCredentialsOutOfBand(
    issuer_url=os.environ.get("ISSUER_URL"), 
    skip_tls_verify=True,
)


_all__ = ["credentials"]