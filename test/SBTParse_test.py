import unittest
import SBTParse as SBT


class SBTParseTest(unittest.TestCase):

    def setUp(self):
        self.sbt_p = SBT.SBTParse()

    def find_sequence_in_sbt(self, sub_sequence: list, main_list: list) -> int:
        """ Assert if the sub_sequence is in the main_list and return the index of the first element of the sub_sequence """
        n = len(sub_sequence)
        for i in range(len(main_list) - n + 1):
            if main_list[i:i + n] == sub_sequence:
                return i
        self.fail(f"Subsequence {sub_sequence} not found in SBT.\n SBT: {main_list}")

    def test_stb_variable_parsing(self):
        code = "x"
        result = self.sbt_p.parse(code)
        expected = ["(", "Name", "(", "x", ")", "x", ")", "Name"]
        self.find_sequence_in_sbt(expected, result)

    def test_stb_constant_parsing(self):
        code = "3"
        result = self.sbt_p.parse(code)
        expected = ["(", "Constant", "(", "3", ")", "3", ")", "Constant"]
        self.find_sequence_in_sbt(expected, result)

    def test_stb_arg_parsing(self):
        code = "def foo(x):\n    pass"
        result = self.sbt_p.parse(code)
        expected = ["(", "arg", "(", "x", ")", "x", ")", "arg"]
        self.find_sequence_in_sbt(expected, result)

    def test_stb_attribute_parsing(self):
        code = "x.size"
        result = self.sbt_p.parse(code)

        start = ["(", "Attribute", "(", "size", ")", "size"]
        end = [")", "Attribute"]

        self.assertLess(self.find_sequence_in_sbt(start, result), self.find_sequence_in_sbt(end, result))

    def test_stb_function_parsing(self):
        code = "def foo(x):\n    pass"
        result = self.sbt_p.parse(code)

        start = ["(", "FunctionDef", "(", "foo", ")", "foo"]
        end = [")", "FunctionDef"]

        self.assertLess(self.find_sequence_in_sbt(start, result), self.find_sequence_in_sbt(end, result))

    def test_stb_async_function_parsing(self):
        code = "async def foo(x):\n    pass"
        result = self.sbt_p.parse(code)

        start = ["(", "AsyncFunctionDef", "(", "foo", ")", "foo"]
        end = [")", "AsyncFunctionDef"]

        self.assertLess(self.find_sequence_in_sbt(start, result), self.find_sequence_in_sbt(end, result))

    def test_stb_class_parsing(self):
        code = "class Foo:\n    pass"
        result = self.sbt_p.parse(code)

        start = ["(", "ClassDef", "(", "Foo", ")", "Foo"]
        end = [")", "ClassDef"]

        self.assertLess(self.find_sequence_in_sbt(start, result), self.find_sequence_in_sbt(end, result))

    def test_stb_keyword_parsing(self):
        code = "print(x, end=' ')"
        result = self.sbt_p.parse(code)

        start = ["(", "keyword", "(", "end", ")", "end"]
        end = [")", "keyword"]

        self.assertLess(self.find_sequence_in_sbt(start, result), self.find_sequence_in_sbt(end, result))

    def test_stb_keyword_parsing_kwargs(self):
        code = "func(**opinion)"
        result = self.sbt_p.parse(code)

        start = ["(", "keyword", "(", "**kwargs", ")", "**kwargs"]
        end = [")", "keyword"]

        self.assertLess(self.find_sequence_in_sbt(start, result), self.find_sequence_in_sbt(end, result))

    def test_stb_alias_parsing(self):
        code = "import math"
        result = self.sbt_p.parse(code)

        start = ["(", "alias", "(", "math", ")", "math"]
        end = [")", "alias"]

        self.assertLess(self.find_sequence_in_sbt(start, result), self.find_sequence_in_sbt(end, result))

    def test_stb_alias_with_asname_parsing(self):
        code = "import math as m"
        result = self.sbt_p.parse(code)

        start = ["(", "alias", "(", "math->m", ")", "math->m"]
        end = [")", "alias"]

        self.assertLess(self.find_sequence_in_sbt(start, result), self.find_sequence_in_sbt(end, result))

    def test_sbt_dfs_recursion(self):
        code = "a + b"
        result = self.sbt_p.parse(code)

        expected = "( BinOp ( Name ( a ) a ) Name ( Add ) Add ( Name ( b ) b ) Name ) BinOp".split()

        self.find_sequence_in_sbt(expected, result)

    def test_sbt_remove_docstring(self):
        code_with_docstring = '''def foo(x):
           """docstring"""
           pass'''
        code_without_docstring = '''def foo(x):
           pass'''

        result_doc_string = self.sbt_p.parse(code_with_docstring)
        result_without_docstring = self.sbt_p.parse(code_without_docstring)

        self.assertEqual(result_doc_string, result_without_docstring)

    def test_sbt_not_remove_a_not_docstring(self):
        code = '''def foo(x):
            x = 5 
            """not a docstring"""'''

        result = self.sbt_p.parse(code)
        self.assertIn("'not a docstring'", result)

    def test_sbt_simple_end_to_end_parsing(self):
        code = "def foo(x):\n    return x + 2"
        result = self.sbt_p.parse(code)

        idx_module = self.find_sequence_in_sbt(["(", "Module"], result)
        idx_func = self.find_sequence_in_sbt(["(", "FunctionDef", "(", "foo", ")", "foo"], result)
        idx_return = self.find_sequence_in_sbt(["(", "Return"], result)
        idx_binop = self.find_sequence_in_sbt(["(", "BinOp"], result)

        self.assertLess(idx_module, idx_func, "Module must appear before FunctionDef")
        self.assertLess(idx_func, idx_return, "FunctionDef must open before Return")
        self.assertLess(idx_return, idx_binop, "Return must open before BinOp")

        self.assertEqual(result[:2], ["(", "Module"], "SBT must start with a Module node")
        self.assertEqual(result[-2:], [")", "Module"], "SBT must end with a Module node")

    def test_sbt_balanced_parentheses(self):
        code = "def foo(x):\n    return x.size + 2"
        result = self.sbt_p.parse(code)

        opened = result.count("(")
        closed = result.count(")")

        self.assertEqual(opened, closed)

    def test_sbt_invalid_syntax_code_should_rais_exception(self):
        code = "def invalid_code("

        with self.assertRaises(SyntaxError) as e:
            self.sbt_p.parse(code)

        self.assertEqual(str(e.exception), "Could not parse 'def invalid_code(', it contains invalid syntax.")

if __name__ == '__main__':
    unittest.main()