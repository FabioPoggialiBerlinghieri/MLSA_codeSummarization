import ast
from typing import Dict, List


class CodeTokenizer:

    def __init__(self, vocabulary: Dict[str, int] = None):
        self.vocabulary = vocabulary

    def set_vocabulary(self, vocabulary: Dict[str, int]) -> None:
        self.vocabulary = vocabulary

    def tokenize(self, text: str) -> List[int]:
        code_tree = ast.parse(text)
        sbt = ASTToSBT().parse(code_tree).split()
        return self.__word2idx(sbt)

    def __word2idx(self, words: List[str]) -> List[int]:
        tokens = []
        for word in words:
            if word not in self.vocabulary:
                if word.split("?")[0] in self.vocabulary:
                    tokens.append(self.vocabulary[word.split("?")[0]])
                else:
                    tokens.append(self.vocabulary["UNK"])
            else:
                tokens.append(self.vocabulary[word])
        return tokens

class ASTToSBT:

    def parse(self, code_tree) -> str:
        return self.__dfs(code_tree)


    def __dfs(self, node) -> str:
        if isinstance(node, ast.Name):
            return self.__node_to_string(type(node), node.id)
        if isinstance(node, ast.Constant):
            return self.__node_to_string(type(node), repr(node.value))
        if isinstance(node, ast.arg):
            return self.__node_to_string(type(node), node.arg)

        if isinstance(node, ast.Attribute):
            node_label = node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_label = node.name
        elif isinstance(node, ast.keyword):
            node_label = node.arg if node.arg else "**kwargs"
        elif isinstance(node, ast.alias):
            node_label = f"{node.name}->{node.asname}" if node.asname else node.name
        else:
            node_label = type(node).__name__

        dfs = self.__node_to_string(type(node), node_label, start=True)

        for child in ast.iter_child_nodes(node):
            dfs += self.__dfs(child)

        dfs += self.__node_to_string(type(node), node_label, start=False)
        return dfs


    def __node_to_string(self, nodeType, name: str, start: bool = True) -> str:
        if nodeType == ast.Name:
            return f" ( Var?{name} ) Var?{name}"
        if nodeType == ast.Constant:
            return f" ( Constant?{name} ) Constant?{name}"
        if nodeType == ast.arg:
            return f" ( Argument?{name} ) Argument?{name}"

        parenthesis = "(" if start else ")"

        if nodeType in (ast.FunctionDef, ast.AsyncFunctionDef):
            return f" {parenthesis} Function?{name}"
        if nodeType == ast.ClassDef:
            return f" {parenthesis} Class?{name}"
        if nodeType == ast.Attribute:
            return f" {parenthesis} Attr?{name}"
        if nodeType == ast.keyword:
            return f" {parenthesis} Kwarg?{name}"
        if nodeType == ast.alias:
            return f" {parenthesis} Import?{name}"

        return f" {parenthesis} {nodeType.__name__}"
