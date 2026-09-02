import ast

class SBT_Parse:

    def parse(self, code_tree) -> list[str]:
        return self.__dfs(ast.parse(code_tree))

    def __remove_docstring(self, node):
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        return node

    def __dfs(self, node) -> list[str]:
        # foglie
        if isinstance(node, ast.Name):
            return ["(", "Name", "(", str(node.id), ")", str(node.id), ")", "Name"]
        if isinstance(node, ast.Constant):
            return ["(", "Constant", "(", repr(node.value), ")", repr(node.value), ")", "Constant"]
        if isinstance(node, ast.arg):
            return ["(", "arg", "(", str(node.arg), ")", str(node.arg), ")", "arg"]

        # nodi strutturali
        node_type = type(node).__name__
        tokens = ["(", node_type]

        # estrazione manuale dei nomi per i nodi che non sono foglie
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

        # se il nodo ha un identificatore, lo aggiungiamo come figlio
        if identifier:
            tokens.extend(["(", identifier, ")", identifier])

        # ricerca in profondita
        for child in ast.iter_child_nodes(node):
            tokens.extend(self.__dfs(child))

        tokens.extend([")", node_type])

        return tokens
