import os
import kfp
from src.utils import credentials


kfp_client = kfp.Client(
    host=os.environ.get("PIPELINES_HOST"),
    verify_ssl=not credentials.skip_tls_verify,
    credentials=credentials,
)

experiments = kfp_client.list_experiments(namespace="team-1")
print(experiments)
