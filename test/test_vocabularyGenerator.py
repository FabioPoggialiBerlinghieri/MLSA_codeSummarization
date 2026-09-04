import unittest
import vocabulary_generator as vg

class TestVocabularyGenerator(unittest.TestCase):

    def setUp(self):
        self.codes = ["a = x*x", "return b"]
        self.vocgen = vg.PythonVocabularyGenerator(self.codes)
        self.special_tokens = self.vocgen.special_tokens
        self.firstCodeTokens = {"x", "a", "Assign", "Module", "(", ")", "Name", "BinOp", "Mult"}

    def test_generate_singleCodeDefaultMaxLength(self):
        self.vocgen.codes = ["a = x*x"]
        expectedElements = self.firstCodeTokens | {"UKN"} | self.special_tokens.keys()
        expectedIndexes = set([i for i in range(len(expectedElements))])

        voc = self.vocgen.generate()

        self.assertEqual(expectedElements, voc.keys())
        self.assertEqual(expectedIndexes, set(voc.values()))
        self.assertEqual(len(expectedElements), len(voc))

        self.assertTrue(voc["("] == len(self.special_tokens) + 1)
        self.assertTrue(voc[")"] == len(self.special_tokens) + 2)

        self.assertTrue(voc[")"] < voc["x"] and voc["("] < voc["x"])
        self.assertTrue(voc["x"] < voc["Assign"])

    def test_generate_singleCodeFixedMaxLength(self):
        self.vocgen.codes = ["x = a + y"]
        self.vocgen.LEN_MAX = 13
        expectedElements = {"(", ")", "Name", "UKN"} | self.special_tokens.keys()
        expectedIndexes = set([i for i in range(len(expectedElements))])

        voc = self.vocgen.generate()

        self.assertEqual(self.vocgen.LEN_MAX, len(voc))
        self.assertEqual(expectedElements, voc.keys())
        self.assertEqual(expectedIndexes, set(voc.values()))

        self.assertTrue(voc["("] == len(self.special_tokens) + 1)
        self.assertTrue(voc[")"] == len(self.special_tokens) + 2)

        self.assertTrue(voc[")"] < voc["Name"] and voc["("] < voc["Name"])

    def test_generate_multipleCode(self):
        secondCodeTokens = {"Return", "b"}
        expectedElements = self.firstCodeTokens | {"UKN"} | secondCodeTokens | self.special_tokens.keys()
        expectedIndexes = set([i for i in range(len(expectedElements))])

        voc = self.vocgen.generate()

        self.assertEqual(expectedElements, voc.keys())
        self.assertEqual(expectedIndexes, set(voc.values()))
        self.assertEqual(len(expectedElements), len(voc))

        self.assertTrue(voc["("] == len(self.special_tokens) + 1)
        self.assertTrue(voc[")"] == len(self.special_tokens) + 2)

        self.assertTrue(voc[")"] < voc["b"] and voc["("] < voc["b"])


    def test_generate_emptyCodeList(self):
        self.vocgen = vg.PythonVocabularyGenerator([])
        self.assertEqual(self.vocgen.special_tokens | {"UKN": len(self.vocgen.special_tokens)}, self.vocgen.generate())

    def test_invalidMaxLength(self):
        self.assertRaisesRegex(ValueError,
                               "Vocabulary max length must be at least 10",
                               lambda: vg.PythonVocabularyGenerator(self.codes, 5))

    def test_get_main_keywords(self):
        expected_special_tokens = {"Name", "Constant", "arg", "FunctionDef", "AsyncFunctionDef",
            "ClassDef", "Attribute", "keyword", "alias"}
        self.assertEqual(expected_special_tokens, set(self.vocgen.get_main_keywords()))


