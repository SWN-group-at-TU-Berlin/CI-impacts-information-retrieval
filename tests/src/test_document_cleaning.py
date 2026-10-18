import json
from unittest import TestCase
from unittest.mock import patch
from my_module import get_user_email

import numpy as np
import pandas as pd

from src import document_cleaning as dc
from src import postprocess as pp
from src import translation_model as tm
from src import utils


def test_extract_citation_info():
    assert dc.extract_citation_info("Koks et al 2022 - Flood impacts") == (
        "Koks ",
        "2022",
        " - Flood impacts",
    )


def test_remove_urls():
    text = "Read https://example.com and www.example.org now."
    assert dc.remove_urls(text) == "Read  and  now."


def test_remove_references():
    text = "Body text.\nReferences\nKoks et al. 2022"
    assert dc.remove_references(text) == "Body text.\nReferences"


def test_remove_figure_references():
    assert dc.remove_figure_references("Roads were damaged (A1).") == (
        "Roads were damaged."
    )


def test_clean_text_normalizes_whitespace_and_punctuation():
    text = '  Some   text  , with spaces  .  " '
    assert dc.clean_text(text) == 'Some text, with spaces."'


def test_sentence_end():
    assert dc.is_sentence_end("Finished.")
    assert dc.is_sentence_end('Finished."')
    assert not dc.is_sentence_end("Still ongoing")


def test_combine_paragraphs():
    assert dc.combine_paragraphs("First paragraph", "second paragraph") == (
        "First paragraph second paragraph"
    )
    assert dc.combine_paragraphs("First paragraph.", "Second paragraph") == (
        "First paragraph.\nSecond paragraph"
    )


def test_split_sentences_preserves_abbreviations():
    text = "Dr. Smith arrived. The road was closed."
    assert tm.split_sentences_keep_sep(text) == [
        "Dr. Smith arrived.",
        " The road was closed.",
    ]


def test_split_long_paragraph_returns_two_parts():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    first, second = tm.split_long_paragraph(text)

    assert first == ["First sentence.", " Second sentence."]
    assert second == [" Third sentence.", " Fourth sentence."]


def test_group_ci_types_assigns_matching_subgroup():
    patterns = pd.DataFrame(
        [
            {
                "pattern": [{"TEXT": {"REGEX": r"road.*"}}],
                "subgroup_name": "roads",
            }
        ]
    )
    data = pd.DataFrame({"asset": ["roads", "railway"]})

    result = pp.group_ci_types(data, "asset", "group", patterns)

    assert result["group"].tolist() == ["roads", None]


def test_postprocess_response_parses_numbered_fields():
    response = json.dumps(
        {
            "infrastructure_type_1": "roads",
            "location_1": "Berlin",
            "damage_1": "blocked",
            "damage_value_1": "NAN",
        }
    )

    result = pp.postprocess_response(response)

    assert result.iloc[0]["infrastructure_type"] == "roads"
    assert result.iloc[0]["location"] == "Berlin"
    assert result.iloc[0]["damage"] == "blocked"


# def test_gen_dict_extract():
#     value = {
#         "countries": [
#             {"name": "Germany"},
#             {"nested": {"name": "France"}},
#         ]
#     }

#     assert list(utils.gen_dict_extract(value, "name")) == ["Germany", "France"]


def test_metrics():
    assert utils.calc_recall(8, 2) == 0.8
    assert utils.calc_precision(8, 2) == 0.8
    assert utils.calc_f1(0.8, 0.8) == 0.8


## Unit test with a mock
## NOTE: "Mocking is the solution when your unit test would call a service that’s outside of your domain." [Source: https://www.signadot.com/blog/why-developers-shouldnt-write-mocks-a-guide-to-modern-testing/]
# Your mocked function (or mocked code in general) acts in a way that it simulates another service (e.g., an API) or some data, etc.
# In other words, we use mocking when we want to replace a function/data etc. with a dummy representation of it.
# For example, if we need an API call for a test function, then we dont want to call the API each time when we run the test.
# For such a case, we mock the API call and provide a Dummy version of the API call 


class TestNominatimCall(TestCase):
    
    @patch('my_module.requests.get')
    def test_get_user_email(self, mock_get):
        # Arrange: set up the fake return value
        mock_get.return_value.json.return_Value = {"email": "test@example.com"}
        
        # Act: call the function
        email = get_user_email(123)
        
        # Assert: check the result
        self.assertEqual(email, "test@example.com")
        mock_get.assert_called_once_with("https://example.com")


def test_supports_flash_attention():
    with patch(
        "src.utils.torch.cuda.get_device_capability",
        return_value=(8, 6),
    ):
        assert utils.supports_flash_attention(0)

    with patch(
        "src.utils.torch.cuda.get_device_capability",
        return_value=(7, 5),
    ):
        assert not utils.supports_flash_attention(0)
