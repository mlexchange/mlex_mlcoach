import hashlib
import logging
import os

import httpx
from humanhash import humanize
from tiled.client import from_uri
from tiled.client.array import ArrayClient
from tiled.client.container import Container
from tiled.queries import Contains

RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")

logger = logging.getLogger(__name__)


class TiledDataLoader:
    def __init__(self, data_tiled_uri, data_tiled_api_key):
        self.data_tiled_uri = data_tiled_uri
        self.data_tiled_api_key = data_tiled_api_key
        self.refresh_data_client()

    def refresh_data_client(self):
        try:
            self.data_client = from_uri(
                self.data_tiled_uri,
                api_key=self.data_tiled_api_key,
                timeout=httpx.Timeout(30.0),
            )
        except Exception as e:
            logger.warning(f"Error connecting to Tiled: {e}")
            self.data_client = None

    def check_dataloader_ready(self):
        """
        Check if the data client is available and ready to be used.
        If base_only is True, only check the base uri.
        """
        if self.data_client is None:
            # Try refreshing once
            self.refresh_data_client()
            return False if self.data_client is None else True
        else:
            try:
                headers = {"Authorization": f"Bearer {self.data_tiled_api_key}"}
                httpx.get(self.data_tiled_uri, headers=headers)
            except Exception as e:
                logger.warning(f"Error connecting to Tiled: {e}")
                return False
        return True

    def get_available_data_names(self):
        """
        Get available data names from the main Tiled container,
        filtered by types that can be processed (Container and ArrayClient)
        """
        if self.data_client is None:
            return []
        data_names = [
            name
            for name in list(self.data_client)
            if isinstance(self.data_client[name], (Container, ArrayClient))
        ]
        return data_names

    def prepare_project_container(self, user, project_name):
        """
        Prepare a project container in the data store
        """
        last_container = self.data_client
        for part in [user, project_name]:
            if part in last_container.keys():
                last_container = last_container[part]
            else:
                last_container = last_container.create_container(key=part)
        return last_container

    def query_by_trimmed_uri(self, trimmed_uri, key, value):
        """
        Query metadata for a trimmed uri
        """
        return self.data_client[trimmed_uri].search(Contains(key, value))

    def get_data_by_trimmed_uri(self, trimmed_uri, slice=None, indx=None):
        """
        Retrieve data by a trimmed uri (not containing the base uri) and slice id/index
        """
        if slice is None and indx is None:
            return self.data_client[trimmed_uri]
        elif slice is not None and indx is None:
            return self.data_client[trimmed_uri][slice]
        else:
            return self.data_client[trimmed_uri].read().iloc[indx]

    def get_metadata_by_trimmed_uri(self, trimmed_uri):
        """
        Retrieve metadata by a trimmed uri (not containing the base uri)
        """
        return self.data_client[trimmed_uri].metadata


tiled_results = TiledDataLoader(
    data_tiled_uri=RESULTS_TILED_URI, data_tiled_api_key=RESULTS_TILED_API_KEY
)


def hash_list_of_strings(strings_list):
    """
    Produces a hash of a list of strings.
    """
    concatenated = "".join(strings_list)
    digest = hashlib.sha256(concatenated.encode("utf-8")).hexdigest()
    return humanize(digest)
