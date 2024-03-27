import logging

from connection import mongo
from connection.github_api import GitHubAPI

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s"
)
log = logging.getLogger(__name__)

log.info("Connecting to MongoDB...")
mo = mongo.Mongo()

log.info("Creating GitHub API Client...")
github_api_client = GitHubAPI()
