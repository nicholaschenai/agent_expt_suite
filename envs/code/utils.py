import ast


def visit_imports(node):
    """
    Visit an AST node and collect module names from import statements.

    Args:
        node (ast.AST): The AST node to visit.

    Returns:
        set: A set of module names imported in the node.
    """
    modules = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            modules.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        modules.append(node.module)
    return set(modules)
