import ast
from typing import Dict, List


class CodeTokenizer:

    def __init__(self, vocabulary: Dict[int, str]):
        self.vocabulary = vocabulary

    def set_vocabulary(self, vocabulary: Dict[int, str]) -> None:
        self.vocabulary = vocabulary

    def tokenize(self, text: str) -> List[int]:
        code_tree = ast.parse(text)
        sbt = ASTToSBT.parse(code_tree).split()
        return self.__word2idx(sbt)

    def __word2idx(self, word: List[str]) -> List[int]:
        return []

class ASTToSBT:

    def parse(self, code_tree) -> str:
        return self.__dfs(code_tree)

    def __dfs(self, node) -> str:
        if isinstance(node, ast.Name):
            return self.__node_to_string(type(node), node.id)
        if isinstance(node, ast.Constant):
            return self.__node_to_string(type(node), str(node.value))

        node_label = getattr(node, "name", getattr(node, "arg", type(node).__name__))
        dfs = self.__node_to_string(type(node), node_label, start=True)

        for child in ast.iter_child_nodes(node):
            dfs += self.__dfs(child)

        dfs += self.__node_to_string(type(node), node_label, start=False)
        return dfs

    def __node_to_string(self, nodeType, name: str, start:bool=True) -> str:
        if nodeType == ast.Name:
            return f" ( Var_{name} ) Var_{name}"
        if nodeType == ast.Constant:
            return f" ( Constant_{name} ) Constant_{name}"

        parenthesis = "(" if start else ")"
        if nodeType == ast.FunctionDef:
            return f" {parenthesis} Function_{name}"
        elif nodeType == ast.ClassDef:
            return f" {parenthesis} Class_{name}"
        return f" {parenthesis} {nodeType.__name__}"

code = """
if x > 0:
    y = 1
else:
    y = 0
"""
print(ASTToSBT().parse(ast.parse(code)))