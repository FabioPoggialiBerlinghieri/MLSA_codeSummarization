import unittest
import SBTParse as SBT

class STBParseTest(unittest.TestCase):

    def setUp(self):
        self.sbt_p = SBT.SBTParse()

    def test_stb_variable_parsing(self):
        expected = "( Module ( Expr ( Name ( x ) x ) Name ) Expr ) Module ".split()
        result = self.sbt_p.parse("x")

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()