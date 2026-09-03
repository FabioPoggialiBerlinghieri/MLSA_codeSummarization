import unittest
import SBTParse as SBT
from codeTokenizer import CodeTokenizer, InvalidUknownIdentifierException, InvalidMainKeywordsException


class SBTParseTest(unittest.TestCase):

    def setUp(self):
        self.cd = CodeTokenizer({'a_c' : 1, 'b_c' : 2, 'c' : 3, 'd' : 4, 'e' : 5, 'a' : 6}, ['a'], uknown_identifier = 'c')

    def test_init_with_wrong_uknown_identifier_should_raise_error(self):

        with self.assertRaises(InvalidUknownIdentifierException) as e:
            CodeTokenizer({'a' : 1, 'b' : '2'}, uknown_identifier = 'c')

        self.assertEqual("uknown identifier must be a key in the vocabulary", str(e.exception))

    def test_init_with_wrong_main_keywords_should_raise_error(self):
        main_keywords = ['x', 'y']
        with self.assertRaises(InvalidMainKeywordsException) as e:
            CodeTokenizer({'a_c' : 1, 'b_c' : 2, 'c' : 3}, main_keywords=main_keywords, uknown_identifier = 'c')

        self.assertEqual(f"missmatch between main_keyword {main_keywords} and vocabulary", str(e.exception))

    def test_init_with_some_wrong_main_keywords_should_raise_error(self):
        main_keywords = ['a', 'y', 'b']
        with self.assertRaises(InvalidMainKeywordsException) as e:
            CodeTokenizer({'a_c' : 1, 'b_c' : 2, 'c' : 3}, main_keywords=main_keywords, uknown_identifier = 'c')

        self.assertEqual(f"missmatch between main_keyword ['y'] and vocabulary", str(e.exception))

    def test_word2idx_with_an_element(self):
        self.assertEqual([4, 5], self.cd.word2idx(['d', 'e']))

    def test_word2idx_mainkeywords(self):
        words = ['a', 'e', '?', 'e', '?']
        expected = [6, 5, 1, 5, 1]

        self.assertEqual(expected, self.cd.word2idx(words))

    def test_word2idx_unkown(self):
        words = ['e', 'e', '?', 'e', '?']
        expected = [5, 5, 3, 5, 3]

        self.assertEqual(expected, self.cd.word2idx(words))

    def test_word2idx_unknown_as_first_element(self):
        self.assertEqual([3], self.cd.word2idx(['?']))

    def test_word2idx_with_multiple_unknowns(self):
        self.assertEqual([3, 3, 3], self.cd.word2idx(['?','?','?']))

    def test_word2idx_empty_list(self):
        self.assertEqual([], self.cd.word2idx([]))

if __name__ == '__main__':
    unittest.main()