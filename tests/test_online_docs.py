import unittest
from unittest.mock import patch
from ml_lab.online_docs import parse_document,fetch_document,official_url


class OnlineDocsTests(unittest.TestCase):
    def test_section_extraction_and_image_resolution_exclude_navigation(self):
        page='''<html><nav><p>Navigation content that must never be included in the requested model documentation.</p></nav>
        <main><section id="ridge"><h2>Ridge regression</h2><p>This model uses a linear prediction function and regularizes the coefficients with an L2 penalty.</p>
        <img src="../_images/ridge.png" alt="Coefficient paths"><script>bad()</script></section>
        <section><p>This text belongs to a different algorithm and must not be included in the requested section.</p></section></main></html>'''
        parsed=parse_document(page,'https://scikit-learn.org/stable/modules/linear_model.html#ridge')
        self.assertEqual(parsed['title'],'Ridge regression')
        self.assertNotIn('different algorithm',parsed['text'])
        self.assertNotIn('Navigation',parsed['text'])
        self.assertEqual(parsed['images'][0]['url'],'https://scikit-learn.org/stable/_images/ridge.png')

    def test_unknown_hosts_and_unsafe_image_urls_are_not_loaded(self):
        self.assertFalse(official_url('http://scikit-learn.org/a'))
        self.assertFalse(official_url('https://scikit-learn.org.attacker.test/a'))
        parsed=parse_document('<main><img src="javascript:alert(1)"><img src="https://other.test/track.png"></main>','https://scikit-learn.org/a')
        self.assertEqual(parsed['images'],[])
        with patch('ml_lab.online_docs.requests.get') as get:
            self.assertIn('error',fetch_document('https://other.test/a'))
            get.assert_not_called()

    def test_network_failure_has_a_nonfatal_result(self):
        import requests
        with patch('ml_lab.online_docs.requests.get',side_effect=requests.Timeout('timeout')):
            self.assertIn('error',fetch_document('https://scikit-learn.org/a'))
