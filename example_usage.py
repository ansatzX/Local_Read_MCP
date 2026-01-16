# Example usage of local_read_mcp server

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp import ClientSession
import asyncio


async def main():
    """Example of using local_read_mcp server."""

    # Configure to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "local_read_mcp.server"],
    )

    print("=" * 50)
    print("Local Read MCP Server Example")
    print("=" * 50)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize to session
            await session.initialize()
            print("\nConnected to local_read_mcp server\n")

            # List available tools
            tools = await session.list_tools()
            print(f"\nAvailable tools: {len(tools)}")
            for tool in tools:
                print(f"   - {tool.name}")

            # Example 1: Read a PDF file
            print("\n" + "-" * 50)
            print("Example 1: Read PDF")
            print("-" * 50)

            pdf_result = await session.call_tool(
                "read_pdf",
                arguments={"file_path": "/path/to/document.pdf"}
            )
            print(f"\nResult: {pdf_result.content[-1].text[:200]}...\n")

            # Example 2: Read a Word document
            print("\n" + "-" * 50)
            print("Example 2: Read Word")
            print("-" * 50)

            word_result = await session.call_tool(
                "read_word",
                arguments={"file_path": "/path/to/document.docx"}
            )
            print(f"\nResult: {word_result.content[-1].text[:200]}...\n")

            # Example 3: Read an Excel file
            print("\n" + "-" * 50)
            print("Example 3: Read Excel")
            print("-" * 50)

            excel_result = await session.call_tool(
                "read_excel",
                arguments={"file_path": "/path/to/spreadsheet.xlsx"}
            )
            print(f"\nResult: {excel_result.content[-1].text[:200]}...\n")

            # Example 4: Read a text file
            print("\n" + "-" * 50)
            print("Example 4: Read Text")
            print("-" * 50)

            text_result = await session.call_tool(
                "read_text",
                arguments={"file_path": "/path/to/file.txt"}
            )
            print(f"\nResult:\n{text_result.content[-1].text}\n")

            # Example 5: Read JSON
            print("\n" + "-" * 50)
            print("Example 5: Read JSON")
            print("-" * 50)

            json_result = await session.call_tool(
                "read_json",
                arguments={"file_path": "/path/to/data.json"}
            )
            print(f"\nResult: {json_result.content[-1].text[:200]}...\n")

            # Example 6: Get supported formats
            print("\n" + "-" * 50)
            print("Example 6: Get Supported Formats")
            print("-" * 50)

            formats = await session.call_tool(
                "get_supported_formats",
                arguments={}
            )
            print(f"\nSupported formats info:\n{formats.content[-1].text}")


if __name__ == "__main__":
    asyncio.run(main())
