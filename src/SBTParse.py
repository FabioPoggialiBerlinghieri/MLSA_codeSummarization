import ast

class SBTParse:

    def parse(self, code: str) -> list[str]:
        """Parse an indented Python source code string and return its AST encoded as an SBT."""
        try:
            ast_code = ast.parse(code)
            return self.__dfs(ast_code)
        except:
            raise SyntaxError(f"Could not parse '{code}', it contains invalid syntax.")

    def __remove_docstring(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef:

        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        return node

    def __dfs(self, node: ast.AST ) -> list[str]:
        # leaf
        if isinstance(node, ast.Name):
            return ["(", "Name", "(", str(node.id), ")", str(node.id), ")", "Name"]
        if isinstance(node, ast.Constant):
            return ["(", "Constant", "(", repr(node.value), ")", repr(node.value), ")", "Constant"]
        if isinstance(node, ast.arg):
            return ["(", "arg", "(", str(node.arg), ")", str(node.arg), ")", "arg"]

        # structured nodes
        node_type = type(node).__name__
        tokens = ["(", node_type]

        # manual extraction of names for non-leaf nodes
        identifier = None
        if isinstance(node, ast.Attribute):
            identifier = node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifier = node.name
            node = self.__remove_docstring(node)
        elif isinstance(node, ast.keyword):
            identifier = node.arg if node.arg else "**kwargs"
        elif isinstance(node, ast.alias):
            identifier = f"{node.name}->{node.asname}" if node.asname else node.name

        # if the nodes has an identifier, we add it as child
        if identifier:
            tokens.extend(["(", identifier, ")", identifier])

        # deep search
        for child in ast.iter_child_nodes(node):
            tokens.extend(self.__dfs(child))

        tokens.extend([")", node_type])

        return tokens


