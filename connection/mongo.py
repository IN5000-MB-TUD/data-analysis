import logging
import os

import pymongo
from pymongo import errors as pe

# Setup logging
log = logging.getLogger(__name__)


class Mongo:
    """
    MongoDB's connection class.
    """

    def __init__(self, connect=False, tz_aware=True):
        """
        Initialize the MongoDB connection.

        :param connect: The connect flag.
        :param tz_aware: The time zone awareness flag.
        """

        host = os.getenv("MONGODB_HOST", "")
        port = os.getenv("MONGODB_PORT", "")
        # add a colon to the port if the port exists
        if port != "":
            port = ":{}".format(port)
        user = os.getenv("MONGODB_USER", "")
        password = os.getenv("MONGODB_PASSWORD", "")
        database = os.getenv("MONGODB_DATABASE", "")
        query_params = os.getenv("MONGODB_QPARAMS", "")
        # add a question mark to the query parameters if they exist.
        if query_params != "":
            query_params = "?{}".format(query_params)

        try:
            self.conn = pymongo.MongoClient(
                "mongodb+srv://{}:{}@{}{}/{}{}".format(
                    user, password, host, port, database, query_params
                ),
                connect=connect,
                tz_aware=tz_aware,
            )
            self.db = self.conn[database]
        except pe.InvalidURI:
            try:
                self.conn = pymongo.MongoClient(
                    "mongodb://{}:{}@{}{}/{}{}".format(
                        user, password, host, port, database, query_params
                    ),
                    connect=connect,
                    tz_aware=tz_aware,
                )
                self.db = self.conn[database]
            except pe.InvalidURI:
                self.conn = pymongo.MongoClient(
                    "mongodb://{}{}/{}{}".format(host, port, database, query_params),
                    connect=connect,
                    tz_aware=tz_aware,
                )
                self.db = self.conn[database]
        except pe.ConnectionFailure as e:
            log.warning(f"Could not connect to MongoDB: {e}")
