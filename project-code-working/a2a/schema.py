
class ToolCall:
    def __init__(self, tool, name, arguments):
        self.tool = tool
        self.name = name
        self.arguments = arguments

class ToolResult:
    def __init__(self, output):
        self.output = output

class ToolDefinition:
    def __init__(self, name, description=None, parameters=None):
        self.name = name
        self.description = description or ""
        self.parameters = parameters or {}
