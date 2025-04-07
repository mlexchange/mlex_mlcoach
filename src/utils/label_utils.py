import logging
import os
from typing import List

from src.utils.data_utils import TiledDataLoader

logging.basicConfig(encoding="utf-8", level=logging.INFO)

LABELS_TILED_URI = os.getenv("LABELS_TILED_URI", "http://localhost:8888")
LABELS_TILED_API_KEY = os.getenv("LABELS_TILED_API_KEY", None)
USER = os.getenv("USER", "mlcoach_user")


class Labels(TiledDataLoader):
    def __init__(self, labels_tiled_uri, labels_tiled_api_key=None):
        super().__init__(labels_tiled_uri, labels_tiled_api_key)

    def get_labeling_events(self, project_name):
        trimmed_uri = self.parse_tiled_url(USER, project_name)
        try:
            labels_client = self.get_data_by_trimmed_uri(trimmed_uri)
        except Exception as e:
            logging.error(f"Error connecting to Tiled: {e}")
            return []

        event_options = []
        for event_id in labels_client.keys():

            expected_url = self.parse_tiled_url(USER, project_name, event_id)
            event_metadata = self.get_metadata_by_trimmed_uri(expected_url)
            tagger_id = event_metadata.get("tagger_id", "Unknown")
            tagging_event_time = event_metadata.get("run_time", "Unknown")

            event_options.append(
                {
                    "label": f"Tagger ID: {tagger_id}, modified: {tagging_event_time}",
                    "value": event_id,
                }
            )
        return event_options

    def get_labeled_indices(
        self, uris: List[str], project_name: str, event_id: str
    ) -> List[int]:
        trimmed_uri = self.parse_tiled_url(USER, project_name, event_id)
        all_labeled_items = self.query_by_trimmed_uri(trimmed_uri, key="uri", value="")
        labeled_uri_set = {
            self.get_metadata_by_trimmed_uri(f"{trimmed_uri}/{item}")["uri"]
            for item in all_labeled_items
        }
        matched_indices = [i for i, uri in enumerate(uris) if uri in labeled_uri_set]
        return sorted(matched_indices)

    def get_label(self, project_name, event_id, uri):
        trimmed_uri = self.parse_tiled_url(USER, project_name, event_id)
        label_uid = list(self.query_by_trimmed_uri(trimmed_uri, key="uri", value=uri))

        if len(label_uid) == 0:
            return "Not labeled"

        try:
            metadata = self.get_metadata_by_trimmed_uri(f"{trimmed_uri}/{label_uid[0]}")
            label = metadata["label"]
        except Exception as e:
            logging.error(f"Error connecting to Tiled: {e}")
            return "Not labeled"
        return label

    @staticmethod
    def parse_tiled_url(user, project_name, event_id=None):
        """
        Parse a Tiled URL to extract the user, project name, and event ID
        Args:
            user:               User name
            project_name:       Project name
            event_id:           Event ID
        Returns:
            parsed_uri:         Parsed URI
        """
        if event_id is None:
            return f"/{user}/{project_name}/labels"
        else:
            return f"/{user}/{project_name}/labels/{event_id}"

    def get_full_tiled_url(self, user, project_name, event_id=None):
        """
        Get the Tiled URL for a given user, project name, and event ID
        Args:
            user:               User name
            project_name:       Project name
            event_id:           Event ID
        Returns:
            tiled_url:          Tiled URL
        """
        tiled_uri = self.data_client.uri
        if event_id is None:
            return f"{tiled_uri}/{user}/{project_name}/labels"
        else:
            return f"{tiled_uri}/{user}/{project_name}/labels/{event_id}"


labels = Labels(
    labels_tiled_uri=LABELS_TILED_URI,
    labels_tiled_api_key=LABELS_TILED_API_KEY,
)
